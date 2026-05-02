"""Run Bedrock-backed WaaT evaluation for workflows that pause for user confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from waat.agent import WaaTSuperAgent
from waat.baseline import PromptWorkflowBaseline
from waat.workflow import load_workflow


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "multiturn"
DEFAULT_WORKFLOW_PATH = ROOT / "waat" / "workflows" / "service_request_multiturn.yaml"
DEFAULT_APPLICATION_INFERENCE_PROFILE = (
    "arn:aws:bedrock:ap-southeast-2:041538338020:application-inference-profile/umjk7k37bjmb"
)


def main() -> None:
    args = _parse_args()
    _load_env_files(ROOT.parent / ".env", ROOT / ".env")
    workflow = load_workflow(args.workflow_path)
    bedrock_verify_ssl: bool | str = args.ca_bundle or not args.no_verify_ssl
    if args.no_verify_ssl:
        _disable_insecure_request_warnings()
    output_dir = args.output_dir or (RESULTS_DIR / "confirmation")
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = WaaTSuperAgent(
        workflow,
        args.workflow_path,
        bedrock_profile=args.profile,
        bedrock_region=args.region,
        bedrock_model_id=args.model_id,
        bedrock_verify_ssl=bedrock_verify_ssl,
    )
    prompt_baseline = PromptWorkflowBaseline(
        workflow,
        args.workflow_path,
        bedrock_profile=args.profile,
        bedrock_region=args.region,
        bedrock_model_id=args.model_id,
        bedrock_verify_ssl=bedrock_verify_ssl,
    )

    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    cases = _multiturn_cases()
    selected_cases = cases[: args.limit] if args.limit else cases
    for index, case in enumerate(selected_cases, start=1):
        _print_progress(index, len(selected_cases), case["case_id"])
        first_turn = agent.run_case(case)
        result = first_turn
        for user_turn in case.get("user_turns", []):
            if result.status != "waiting_for_user" or result.session is None:
                break
            result = agent.resume_case(result.session, case, user_turn)

        baseline_result = prompt_baseline.run_case(case)
        rows.append(
            {
                "case_id": case["case_id"],
                "customer_id": case["customer_id"],
                "request_text": case["request_text"],
                "status": result.status,
                "expected_terminal_state": case["expected_terminal_state"],
                "actual_terminal_state": result.terminal_state,
                "expected_path": case["expected_path"],
                "actual_path": result.actual_path,
                "transition_accuracy": result.transition_accuracy,
                "path_match": result.path_match,
                "terminal_match": result.terminal_match,
                "prompt_baseline_terminal_state": baseline_result.terminal_state,
                "prompt_baseline_path": baseline_result.actual_path,
                "prompt_baseline_transition_accuracy": baseline_result.transition_accuracy,
                "prompt_baseline_path_match": baseline_result.path_match,
                "prompt_baseline_terminal_match": baseline_result.terminal_match,
                "prompt_baseline_tokens": baseline_result.tokens,
                "total_tokens": result.total_tokens,
                "mean_tokens_per_step": result.mean_tokens_per_step,
            }
        )
        traces[case["case_id"]] = {
            "case": case,
            "first_turn": _serialise_result(first_turn),
            "final_result": _serialise_result(result),
            "prompt_baseline_trace": baseline_result.trace,
        }
        _write_artifacts(output_dir, rows, traces)

    print()
    print(f"Wrote {len(rows)} multiturn rows and aggregate artifacts to {output_dir}")


def _multiturn_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "MT-01",
            "customer_id": "SVC-MT-001",
            "request_text": "Please upgrade my service to the next speed tier from next month.",
            "expected_terminal_state": "REQUEST_COMPLETE",
            "expected_path": [
                "CLASSIFY_REQUEST",
                "HANDLE_SERVICE_CHANGE",
                "CONFIRM_SERVICE_CHANGE",
                "SUBMIT_SERVICE_ORDER",
                "SEND_CUSTOMER_CONFIRMATION",
                "REQUEST_COMPLETE",
            ],
            "metadata": {"category": "service", "change_type": "upgrade"},
            "user_turns": [{"confirmed": True}],
        },
        {
            "case_id": "MT-02",
            "customer_id": "SVC-MT-002",
            "request_text": "Please prepare a service upgrade, but do not submit it unless I approve.",
            "expected_terminal_state": "REQUEST_COMPLETE",
            "expected_path": [
                "CLASSIFY_REQUEST",
                "HANDLE_SERVICE_CHANGE",
                "CONFIRM_SERVICE_CHANGE",
                "REQUEST_COMPLETE",
            ],
            "metadata": {"category": "service", "change_type": "upgrade"},
            "user_turns": [{"confirmed": False}],
        },
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_cases": 0}
    return {
        "num_cases": len(rows),
        "transition_accuracy_pct": round(100 * sum(row["transition_accuracy"] for row in rows) / len(rows), 2),
        "path_accuracy_pct": round(100 * sum(1 for row in rows if row["path_match"]) / len(rows), 2),
        "terminal_state_accuracy_pct": round(100 * sum(1 for row in rows if row["terminal_match"]) / len(rows), 2),
        "prompt_baseline_transition_accuracy_pct": round(
            100 * sum(row["prompt_baseline_transition_accuracy"] for row in rows) / len(rows),
            2,
        ),
        "prompt_baseline_path_accuracy_pct": round(
            100 * sum(1 for row in rows if row["prompt_baseline_path_match"]) / len(rows),
            2,
        ),
        "prompt_baseline_terminal_state_accuracy_pct": round(
            100 * sum(1 for row in rows if row["prompt_baseline_terminal_match"]) / len(rows),
            2,
        ),
        "mean_tokens_per_case_waat": round(sum(row["total_tokens"] for row in rows) / len(rows), 2),
        "mean_prompt_baseline_tokens_per_case": round(
            sum(row["prompt_baseline_tokens"] for row in rows) / len(rows),
            2,
        ),
        "waat_vs_baseline_token_delta_pct": round(
            100
            * (
                (sum(row["total_tokens"] for row in rows) / sum(row["prompt_baseline_tokens"] for row in rows))
                - 1
            ),
            2,
        ),
    }


def _serialise_result(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "actual_path": result.actual_path,
        "terminal_state": result.terminal_state,
        "message_to_user": result.message_to_user,
        "pending_user_request": result.pending_user_request,
        "trace": result.trace,
        "total_tokens": result.total_tokens,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bedrock-backed multiturn WaaT evaluation.")
    parser.add_argument("--profile", help="AWS profile name, for example an SSO profile.")
    parser.add_argument("--workflow-path", type=Path, default=DEFAULT_WORKFLOW_PATH, help="Workflow YAML file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated artifacts. Defaults to results/multiturn/confirmation.",
    )
    parser.add_argument("--region", default="ap-southeast-2", help="AWS Bedrock runtime region.")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_APPLICATION_INFERENCE_PROFILE),
        help="Amazon Bedrock model ID, inference profile ID, or application inference profile ARN.",
    )
    parser.add_argument("--ca-bundle", help="Path to a corporate CA bundle PEM file for AWS SSL verification.")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable AWS SSL verification for smoke tests.")
    parser.add_argument("--limit", type=int, help="Run only the first N multiturn cases.")
    return parser.parse_args()


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "customer_id",
        "request_text",
        "status",
        "expected_terminal_state",
        "actual_terminal_state",
        "expected_path",
        "actual_path",
        "transition_accuracy",
        "path_match",
        "terminal_match",
        "prompt_baseline_terminal_state",
        "prompt_baseline_path",
        "prompt_baseline_transition_accuracy",
        "prompt_baseline_path_match",
        "prompt_baseline_terminal_match",
        "prompt_baseline_tokens",
        "total_tokens",
        "mean_tokens_per_step",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialised = dict(row)
            for key in ("expected_path", "actual_path", "prompt_baseline_path"):
                serialised[key] = json.dumps(serialised[key])
            writer.writerow(serialised)


def _write_artifacts(output_dir: Path, rows: list[dict[str, Any]], traces: dict[str, Any]) -> None:
    summary = _summary(rows)
    _write_csv(output_dir / "results_table.csv", rows)
    (output_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "sample_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")


def _disable_insecure_request_warnings() -> None:
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


def _print_progress(index: int, total: int, case_id: str) -> None:
    width = 30
    filled = int(width * index / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {index:02d}/{total} {case_id}", end="", flush=True)


if __name__ == "__main__":
    main()
