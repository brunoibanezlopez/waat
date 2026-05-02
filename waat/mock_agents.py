"""Deterministic mock utility agents used by the WaaT evaluation."""

from __future__ import annotations

from typing import Any

from .synthetic_data import expected_path_for_case


_CLASSIFIER_OVERRIDES = {
    # Realistic failure modes: complaint language causes premature complaint routing,
    # ambiguous edge cases are over-classified instead of immediately escalated.
    "TC-03": "HANDLE_COMPLAINT",
    "TC-06": "HANDLE_COMPLAINT",
    "TC-25": "HANDLE_ACCOUNT_QUERY",
    "TC-29": "HANDLE_SERVICE_CHANGE",
}


def call_classifier_agent(case: dict[str, Any], state_id: str) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    next_state = _CLASSIFIER_OVERRIDES.get(case["case_id"]) or _state_for_category(metadata.get("category", "edge"))
    label_by_state = {
        "HANDLE_ACCOUNT_QUERY": "account_query",
        "HANDLE_SERVICE_CHANGE": "service_change",
        "HANDLE_COMPLAINT": "complaint",
        "REQUEST_ESCALATED": "unclassifiable_or_sensitive",
    }
    ambiguous = case["case_id"] in _CLASSIFIER_OVERRIDES
    return {
        "agent": "classifier",
        "classification": label_by_state[next_state],
        "confidence": 0.58 if ambiguous else (0.94 if next_state != "REQUEST_ESCALATED" else 0.41),
        "rationale": (
            f"Request text contains competing cues; selecting {label_by_state[next_state]} as the most likely route."
            if ambiguous
            else f"Request text maps to {label_by_state[next_state]} based on category cues."
        ),
        "ambiguous_signal": ambiguous,
        "recommended_next_state": next_state,
    }


def call_account_agent(case: dict[str, Any], state_id: str) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    query_type = metadata.get("query_type", "unknown")
    if metadata.get("category") == "edge" or query_type in {"data_unavailable", "ownership"}:
        next_state = "REQUEST_ESCALATED"
    elif query_type in {"payment_status", "balance_dispute"}:
        next_state = "HANDLE_COMPLAINT"
    else:
        next_state = "REQUEST_COMPLETE"
    return {
        "agent": "account",
        "query_type": query_type,
        "account_summary": "Recent balance, payment status, and contact details retrieved.",
        "customer_satisfied": next_state == "REQUEST_COMPLETE",
        "data_available": next_state != "REQUEST_ESCALATED",
        "recommended_next_state": next_state,
    }


def call_service_agent(case: dict[str, Any], state_id: str) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    change_type = metadata.get("change_type", "unknown")
    if metadata.get("category") == "edge":
        next_state = "REQUEST_COMPLETE" if metadata.get("edge_type") == "ambiguous" else "REQUEST_ESCALATED"
    elif change_type in {"relocation", "custom_design"}:
        next_state = "REQUEST_ESCALATED"
    elif change_type == "contract_exception":
        # Simulated utility-agent miss: treats emergency upgrade as processable.
        next_state = "REQUEST_COMPLETE"
    else:
        next_state = "REQUEST_COMPLETE"
    return {
        "agent": "service",
        "change_type": change_type,
        "completed": next_state == "REQUEST_COMPLETE",
        "manual_approval_required": next_state == "REQUEST_ESCALATED",
        "confirmation": "Change submitted to the service platform." if next_state == "REQUEST_COMPLETE" else "",
        "recommended_next_state": next_state,
    }


def call_complaint_agent(case: dict[str, Any], state_id: str) -> dict[str, Any]:
    complaint_type = case.get("metadata", {}).get("complaint_type", "general")
    next_state = "REQUEST_ESCALATED" if complaint_type in {"billing_dispute", "regulatory", "unresolved"} else "REQUEST_COMPLETE"
    return {
        "agent": "complaint",
        "sentiment": "highly_negative" if next_state == "REQUEST_ESCALATED" else "recoverable",
        "resolution_offered": next_state == "REQUEST_COMPLETE",
        "customer_accepted": next_state == "REQUEST_COMPLETE",
        "escalation_reason": "Customer remains dissatisfied or issue requires specialist review."
        if next_state == "REQUEST_ESCALATED"
        else "",
        "recommended_next_state": next_state,
    }


def execute_action(
    action: str,
    case: dict[str, Any],
    state_id: str,
    state_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dispatch = {
        "call_classifier_agent": call_classifier_agent,
        "call_account_agent": call_account_agent,
        "call_service_agent": call_service_agent,
        "call_complaint_agent": call_complaint_agent,
    }
    if action in dispatch:
        return dispatch[action](case, state_id)
    return call_domain_workflow_agent(case, state_id, state_spec or {})


def call_domain_workflow_agent(case: dict[str, Any], state_id: str, state_spec: dict[str, Any]) -> dict[str, Any]:
    """Domain-shaped mock utility agent for generated telecom workflows."""
    allowed_targets = [transition["to"] for transition in state_spec.get("transitions", [])]
    next_state = _next_policy_state(case, state_id, allowed_targets)
    return {
        "agent": "domain_workflow",
        "state_id": state_id,
        "case_category": case.get("metadata", {}).get("category"),
        "case_type": _case_type(case),
        "allowed_targets": allowed_targets,
        "recommended_next_state": next_state,
        "rationale": f"{state_id} completed using case metadata and operational policy; next step is {next_state}.",
    }


def _state_for_category(category: str) -> str:
    return {
        "account": "HANDLE_ACCOUNT_QUERY",
        "service": "HANDLE_SERVICE_CHANGE",
        "complaint": "HANDLE_COMPLAINT",
        "edge": "REQUEST_ESCALATED",
    }.get(category, "REQUEST_ESCALATED")


def _next_policy_state(case: dict[str, Any], state_id: str, allowed_targets: list[str]) -> str:
    for workflow_nodes in (100, 50, 20, 6):
        path = expected_path_for_case(case, workflow_nodes) if workflow_nodes != 6 else case["expected_path"]
        if state_id in path:
            index = path.index(state_id)
            if index + 1 < len(path) and path[index + 1] in allowed_targets:
                return path[index + 1]
    return allowed_targets[0] if allowed_targets else "REQUEST_ESCALATED"


def _case_type(case: dict[str, Any]) -> str:
    metadata = case.get("metadata", {})
    for key in ("query_type", "change_type", "complaint_type", "edge_type"):
        if key in metadata:
            return str(metadata[key])
    return "unknown"
