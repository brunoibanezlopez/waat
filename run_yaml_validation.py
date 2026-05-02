"""Validate WaaT workflow YAML files for structural and semantic quality."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml

from waat.bedrock_client import BedrockClaudeClient
from waat.workflow import Workflow, load_workflow


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "validation"
DEFAULT_APPLICATION_INFERENCE_PROFILE = (
    "arn:aws:bedrock:ap-southeast-2:041538338020:application-inference-profile/umjk7k37bjmb"
)
DEFAULT_WORKFLOWS = [
    ROOT / "waat" / "workflows" / "generated" / "service_request_6_nodes.yaml",
    ROOT / "waat" / "workflows" / "generated" / "service_request_20_nodes.yaml",
    ROOT / "waat" / "workflows" / "generated" / "service_request_50_nodes.yaml",
    ROOT / "waat" / "workflows" / "service_request_multiturn.yaml",
]


def main() -> None:
    args = _parse_args()
    _load_env_files(ROOT.parent / ".env", ROOT / ".env")
    bedrock_verify_ssl: bool | str = args.ca_bundle or not args.no_verify_ssl
    if args.no_verify_ssl:
        _disable_insecure_request_warnings()

    client = BedrockClaudeClient(
        model_id=args.model_id,
        region_name=args.region,
        profile_name=args.profile,
        verify_ssl=bedrock_verify_ssl,
        max_retries=args.max_retries,
    )

    workflow_paths = args.workflow_path or DEFAULT_WORKFLOWS
    for workflow_path in workflow_paths:
        label = args.label or _workflow_label(workflow_path)
        output_dir = args.output_dir / label
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Validating {workflow_path} -> {output_dir}")
        report = validate_workflow_yaml(workflow_path, client, max_states=args.limit_states)
        _write_artifacts(output_dir, report)
        print(
            f"  structural={len(report['structural_issues'])} issues, "
            f"semantic={len(report['semantic_findings'])} states, "
            f"max ambiguity={report['summary']['max_ambiguity_score']}"
        )


def validate_workflow_yaml(
    workflow_path: Path,
    client: BedrockClaudeClient,
    max_states: int | None = None,
) -> dict[str, Any]:
    raw = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    structural_issues = _structural_issues(raw)
    workflow = load_workflow(workflow_path)
    semantic_findings = _semantic_findings(workflow, client, max_states=max_states)
    return {
        "workflow_path": str(workflow_path),
        "workflow": workflow.name,
        "summary": _summary(workflow, structural_issues, semantic_findings),
        "structural_issues": structural_issues,
        "semantic_findings": semantic_findings,
    }


def _structural_issues(raw: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required = {"workflow", "initial_state", "terminal_states", "states"}
    missing = required - set(raw or {})
    for key in sorted(missing):
        issues.append({"type": "missing_required_key", "severity": "error", "message": f"Missing key: {key}"})
    if missing:
        return issues

    states = raw.get("states") or {}
    terminal_states = set(raw.get("terminal_states") or [])
    initial_state = raw.get("initial_state")
    if initial_state not in states:
        issues.append(
            {
                "type": "invalid_initial_state",
                "severity": "error",
                "state_id": initial_state,
                "message": "Initial state is not defined in states.",
            }
        )

    for state_id, state in states.items():
        transitions = state.get("transitions")
        if transitions is None:
            issues.append(
                {
                    "type": "missing_transitions",
                    "severity": "error",
                    "state_id": state_id,
                    "message": "State is missing transitions.",
                }
            )
            continue
        if not transitions and state_id not in terminal_states:
            issues.append(
                {
                    "type": "dead_end_state",
                    "severity": "error",
                    "state_id": state_id,
                    "message": "State has no transitions and is not declared terminal.",
                }
            )
        for transition in transitions:
            target = transition.get("to")
            if target not in states and target not in terminal_states:
                issues.append(
                    {
                        "type": "undefined_transition_target",
                        "severity": "error",
                        "state_id": state_id,
                        "target": target,
                        "message": f"Transition target {target!r} is not a state or terminal state.",
                    }
                )

    reachable = _reachable_states(initial_state, states, terminal_states)
    for state_id in sorted(set(states) - reachable):
        issues.append(
            {
                "type": "unreachable_state",
                "severity": "warning",
                "state_id": state_id,
                "message": "State is not reachable from the initial state.",
            }
        )

    for state_id in sorted(states):
        if state_id not in terminal_states and not _can_reach_terminal(state_id, states, terminal_states):
            issues.append(
                {
                    "type": "no_terminal_path",
                    "severity": "error",
                    "state_id": state_id,
                    "message": "State has no path to a terminal state.",
                }
            )
    return issues


def _reachable_states(initial_state: str, states: dict[str, Any], terminal_states: set[str]) -> set[str]:
    if initial_state not in states:
        return set()
    seen: set[str] = set()
    stack = [initial_state]
    while stack:
        state_id = stack.pop()
        if state_id in seen or state_id in terminal_states:
            continue
        seen.add(state_id)
        stack.extend(transition.get("to") for transition in states[state_id].get("transitions", []))
    return seen


def _can_reach_terminal(state_id: str, states: dict[str, Any], terminal_states: set[str]) -> bool:
    seen: set[str] = set()
    stack = [state_id]
    while stack:
        current = stack.pop()
        if current in terminal_states:
            return True
        if current in seen or current not in states:
            continue
        seen.add(current)
        stack.extend(transition.get("to") for transition in states[current].get("transitions", []))
    return False


def _semantic_findings(
    workflow: Workflow,
    client: BedrockClaudeClient,
    max_states: int | None = None,
) -> list[dict[str, Any]]:
    states = [
        (state_id, state)
        for state_id, state in workflow.states.items()
        if len(state.get("transitions", [])) >= 2
    ]
    if max_states:
        states = states[:max_states]

    findings: list[dict[str, Any]] = []
    for index, (state_id, state) in enumerate(states, start=1):
        print(f"  [{index}/{len(states)}] semantic check: {state_id}")
        findings.append(_evaluate_state_semantics(workflow, state_id, state, client))
    return findings


def _evaluate_state_semantics(
    workflow: Workflow,
    state_id: str,
    state: dict[str, Any],
    client: BedrockClaudeClient,
) -> dict[str, Any]:
    system_prompt = (
        "You are evaluating a YAML workflow state before deployment. "
        "Assess only the outgoing transition conditions for this single state. "
        "Use three checks from the WaaT paper: overlap, gap, and vagueness. "
        "Overlap means two or more transition conditions could be satisfied by the same realistic action result. "
        "Gap means realistic action results exist that none of the transitions cover. "
        "Vagueness means a condition is too imprecise to distinguish reliably from adjacent conditions. "
        "Return strict JSON only with keys: ambiguity_score, overlap, gap, vagueness, issues, recommended_rewrite. "
        "ambiguity_score must be an integer from 0 to 100. "
        "overlap, gap, and vagueness must each be objects with keys score and explanation; "
        "each score must be an integer from 0 to 100. "
        "issues must be a list of objects with keys type, severity, transitions, explanation. "
        "recommended_rewrite must be a list of transition objects with keys to and condition."
    )
    payload = {
        "workflow": workflow.name,
        "state_id": state_id,
        "action": state.get("action"),
        "action_params": state.get("action_params", {}),
        "transitions": state.get("transitions", []),
        "terminal_states": sorted(workflow.terminal_states),
    }
    parsed, response = client.converse_json(system_prompt, payload, max_tokens=1600)
    return _normalise_semantic_result(state_id, state, parsed, response.total_tokens)


def _normalise_semantic_result(
    state_id: str,
    state: dict[str, Any],
    parsed: dict[str, Any],
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "action": state.get("action"),
        "transitions": state.get("transitions", []),
        "ambiguity_score": _int_between(parsed.get("ambiguity_score"), 0, 100),
        "overlap": _dimension(parsed.get("overlap")),
        "gap": _dimension(parsed.get("gap")),
        "vagueness": _dimension(parsed.get("vagueness")),
        "issues": parsed.get("issues") if isinstance(parsed.get("issues"), list) else [],
        "recommended_rewrite": (
            parsed.get("recommended_rewrite") if isinstance(parsed.get("recommended_rewrite"), list) else []
        ),
        "tokens": total_tokens,
    }


def _dimension(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"score": 0, "explanation": ""}
    return {
        "score": _int_between(value.get("score"), 0, 100),
        "explanation": str(value.get("explanation", "")),
    }


def _int_between(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, number))


def _summary(
    workflow: Workflow,
    structural_issues: list[dict[str, Any]],
    semantic_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    high_ambiguity = [finding for finding in semantic_findings if finding["ambiguity_score"] >= 60]
    medium_ambiguity = [
        finding for finding in semantic_findings if 30 <= finding["ambiguity_score"] < 60
    ]
    return {
        "workflow": workflow.name,
        "num_states": len(workflow.states),
        "num_terminal_states": len(workflow.terminal_states),
        "num_structural_issues": len(structural_issues),
        "num_semantic_states_checked": len(semantic_findings),
        "max_ambiguity_score": max((finding["ambiguity_score"] for finding in semantic_findings), default=0),
        "mean_ambiguity_score": round(
            sum(finding["ambiguity_score"] for finding in semantic_findings) / len(semantic_findings),
            2,
        )
        if semantic_findings
        else 0,
        "num_high_ambiguity_states": len(high_ambiguity),
        "num_medium_ambiguity_states": len(medium_ambiguity),
        "high_ambiguity_states": [finding["state_id"] for finding in high_ambiguity],
        "medium_ambiguity_states": [finding["state_id"] for finding in medium_ambiguity],
        "total_semantic_tokens": sum(finding["tokens"] for finding in semantic_findings),
    }


def _write_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report["summary"], indent=2), encoding="utf-8")
    _write_semantic_csv(output_dir / "semantic_findings.csv", report["semantic_findings"])
    _write_structural_csv(output_dir / "structural_issues.csv", report["structural_issues"])


def _write_semantic_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    fields = [
        "state_id",
        "action",
        "ambiguity_score",
        "overlap_score",
        "gap_score",
        "vagueness_score",
        "num_issues",
        "issues",
        "recommended_rewrite",
        "tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "state_id": finding["state_id"],
                    "action": finding["action"],
                    "ambiguity_score": finding["ambiguity_score"],
                    "overlap_score": finding["overlap"]["score"],
                    "gap_score": finding["gap"]["score"],
                    "vagueness_score": finding["vagueness"]["score"],
                    "num_issues": len(finding["issues"]),
                    "issues": json.dumps(finding["issues"]),
                    "recommended_rewrite": json.dumps(finding["recommended_rewrite"]),
                    "tokens": finding["tokens"],
                }
            )


def _write_structural_csv(path: Path, issues: list[dict[str, Any]]) -> None:
    fields = ["type", "severity", "state_id", "target", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            writer.writerow({field: issue.get(field, "") for field in fields})


def _workflow_label(path: Path) -> str:
    name = path.stem
    if name == "service_request_multiturn":
        return "multiturn"
    if name.startswith("service_request_") and name.endswith("_nodes"):
        return name.removeprefix("service_request_")
    return name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate WaaT workflow YAML conditions with Bedrock.")
    parser.add_argument(
        "--workflow-path",
        type=Path,
        action="append",
        help="Workflow YAML file to validate. May be provided multiple times. Defaults to active benchmark workflows.",
    )
    parser.add_argument("--label", help="Output subfolder label. Only use with a single workflow.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Validation results directory.")
    parser.add_argument("--profile", help="AWS profile name, for example an SSO profile.")
    parser.add_argument("--region", default="ap-southeast-2", help="AWS Bedrock runtime region.")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_APPLICATION_INFERENCE_PROFILE),
        help="Amazon Bedrock model ID, inference profile ID, or application inference profile ARN.",
    )
    parser.add_argument("--ca-bundle", help="Path to a corporate CA bundle PEM file for AWS SSL verification.")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable AWS SSL verification for local use.")
    parser.add_argument("--max-retries", type=int, default=3, help="Bedrock retries per semantic state check.")
    parser.add_argument("--limit-states", type=int, help="Validate only the first N branch states for smoke tests.")
    args = parser.parse_args()
    if args.label and args.workflow_path and len(args.workflow_path) > 1:
        parser.error("--label can only be used with one --workflow-path")
    return args


def _load_env_files(*paths: Path) -> None:
    key_map = {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "AWS_SESSION_TOKEN",
        "aws_region": "AWS_REGION",
        "aws_default_region": "AWS_DEFAULT_REGION",
    }
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_key = key_map.get(key.strip().lower(), key.strip())
            os.environ.setdefault(env_key, value.strip().strip('"').strip("'"))


def _disable_insecure_request_warnings() -> None:
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


if __name__ == "__main__":
    main()
