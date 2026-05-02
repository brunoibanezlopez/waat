"""Run the WaaT evaluation and write paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

from waat.agent import WaaTSuperAgent
from waat.baseline import PromptWorkflowBaseline
from waat.evaluator import ReasoningEvaluator, aggregate_results
from waat.synthetic_data import adapt_expected_paths, generate_test_cases
from waat.workflow import load_workflow
from waat.workflow_variants import workflow_node_count


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "runs"
DEFAULT_WORKFLOW_PATH = ROOT / "waat" / "workflows" / "service_request.yaml"
DEFAULT_APPLICATION_INFERENCE_PROFILE = (
    "arn:aws:bedrock:ap-southeast-2:041538338020:application-inference-profile/umjk7k37bjmb"
)


def main() -> None:
    args = _parse_args()
    _load_env_files(ROOT.parent / ".env", ROOT / ".env")
    random.seed(42)
    workflow = load_workflow(args.workflow_path)
    workflow_nodes = _count_workflow_nodes(args.workflow_path)
    cases = adapt_expected_paths(generate_test_cases(), workflow_nodes)
    bedrock_verify_ssl: bool | str = args.ca_bundle or not args.no_verify_ssl
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
    evaluator = ReasoningEvaluator(
        bedrock_profile=args.profile,
        bedrock_region=args.region,
        bedrock_model_id=args.model_id,
        bedrock_verify_ssl=bedrock_verify_ssl,
    )

    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    representatives = {"account": None, "service": None, "complaint": None}

    selected_cases = cases[: args.limit] if args.limit else cases
    for index, case in enumerate(selected_cases, start=1):
        _print_progress(index, len(selected_cases), case["case_id"])
        result = agent.run_case(case)
        baseline_result = prompt_baseline.run_case(case)
        reasoning_scores = evaluator.score_trace(result.trace)
        category = case["metadata"]["category"]
        if category in representatives and representatives[category] is None:
            representatives[category] = case["case_id"]
            traces[case["case_id"]] = {
                "case": case,
                "actual_path": result.actual_path,
                "terminal_state": result.terminal_state,
                "trace": result.trace,
                "reasoning_scores": reasoning_scores,
            }

        rows.append(
            {
                "case_id": case["case_id"],
                "customer_id": case["customer_id"],
                "category": category,
                "request_text": case["request_text"],
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
                "reasoning_scores": reasoning_scores,
                "mean_reasoning_score": sum(reasoning_scores) / len(reasoning_scores),
                "total_tokens": result.total_tokens,
                "mean_tokens_per_step": result.mean_tokens_per_step,
            }
        )

    print()
    summary = aggregate_results(rows, workflow_nodes=workflow_nodes)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / f"{args.output_prefix}_results_table.csv", rows)
    (output_dir / f"{args.output_prefix}_summary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / f"{args.output_prefix}_sample_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    (output_dir / f"{args.output_prefix}_results_table_latex.tex").write_text(_latex_table(summary), encoding="utf-8")
    print(f"Wrote {len(rows)} rows and aggregate artifacts to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Claude Sonnet WaaT evaluation on Amazon Bedrock.")
    parser.add_argument("--profile", help="AWS profile name, for example an SSO profile.")
    parser.add_argument("--workflow-path", type=Path, default=DEFAULT_WORKFLOW_PATH, help="Workflow YAML file.")
    parser.add_argument("--output-prefix", default="aws", help="Prefix for generated result artifacts.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Directory for generated artifacts.")
    parser.add_argument("--region", default="ap-southeast-2", help="AWS Bedrock runtime region.")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_APPLICATION_INFERENCE_PROFILE),
        help="Amazon Bedrock model ID, inference profile ID, or application inference profile ARN.",
    )
    parser.add_argument("--ca-bundle", help="Path to a corporate CA bundle PEM file for AWS SSL verification.")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable AWS SSL verification for local smoke tests only.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N synthetic cases.")
    return parser.parse_args()


def _count_workflow_nodes(path: Path) -> int:
    import yaml

    return workflow_node_count(yaml.safe_load(path.read_text(encoding="utf-8")))


def _load_env_files(*paths: Path) -> None:
    """Load simple .env files without printing or persisting secrets."""
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
            key = key.strip()
            env_key = key_map.get(key.lower(), key)
            os.environ.setdefault(env_key, value.strip().strip('"').strip("'"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "customer_id",
        "category",
        "request_text",
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
        "reasoning_scores",
        "mean_reasoning_score",
        "total_tokens",
        "mean_tokens_per_step",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialised = dict(row)
            for key in ("expected_path", "actual_path", "prompt_baseline_path", "reasoning_scores"):
                serialised[key] = json.dumps(serialised[key])
            writer.writerow(serialised)


def _latex_table(summary: dict[str, Any]) -> str:
    return rf"""\begin{{table}}[t]
\centering
\caption{{Aggregate WaaT evaluation results across 30 synthetic service requests.}}
\label{{tab:waat-results}}
\begin{{tabular}}{{lc}}
\hline
\textbf{{Metric}} & \textbf{{Result}} \\
\hline
Workflow nodes & {summary.get("workflow_nodes", 0)} \\
WaaT transition accuracy & {summary["transition_accuracy_pct"]:.2f}\% \\
Prompt baseline transition accuracy & {summary["prompt_baseline_transition_accuracy_pct"]:.2f}\% \\
WaaT path accuracy & {summary["path_accuracy_pct"]:.2f}\% \\
Prompt baseline path accuracy & {summary["prompt_baseline_path_accuracy_pct"]:.2f}\% \\
WaaT terminal state accuracy & {summary["terminal_state_accuracy_pct"]:.2f}\% \\
Prompt baseline terminal accuracy & {summary["prompt_baseline_terminal_state_accuracy_pct"]:.2f}\% \\
Mean reasoning quality & {summary["mean_reasoning_quality"]:.2f} / 5 \\
Mean WaaT tokens / case & {summary["mean_tokens_per_case_waat"]:.2f} \\
Mean WaaT tokens / step & {summary["mean_tokens_per_step_waat"]:.2f} \\
Mean prompt baseline tokens / case & {summary["mean_prompt_baseline_tokens_per_case"]:.2f} \\
Case-level token reduction & {summary["case_token_reduction_pct"]:.2f}\% \\
\hline
\end{{tabular}}
\end{{table}}
"""


def _print_progress(index: int, total: int, case_id: str) -> None:
    width = 30
    filled = int(width * index / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {index:02d}/{total} {case_id}", end="", flush=True)


if __name__ == "__main__":
    main()
