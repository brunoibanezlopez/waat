"""Run the WaaT evaluation and write paper-ready artifacts."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from waat.agent import WaaTSuperAgent
from waat.baseline import PromptWorkflowBaseline
from waat.evaluator import ReasoningEvaluator, aggregate_results
from waat.synthetic_data import generate_test_cases
from waat.workflow import load_workflow


ROOT = Path(__file__).resolve().parent
WORKFLOW_PATH = ROOT / "waat" / "workflows" / "service_request.yaml"


def main() -> None:
    random.seed(42)
    workflow = load_workflow(WORKFLOW_PATH)
    cases = generate_test_cases()
    agent = WaaTSuperAgent(workflow, WORKFLOW_PATH, use_anthropic=False, seed=42)
    prompt_baseline = PromptWorkflowBaseline(workflow, WORKFLOW_PATH)
    evaluator = ReasoningEvaluator(use_anthropic=False, seed=42)

    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    representatives = {"account": None, "service": None, "complaint": None}

    for index, case in enumerate(cases, start=1):
        _print_progress(index, len(cases), case["case_id"])
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
                "baseline_tokens": result.baseline_tokens,
                "mean_tokens_per_step": result.mean_tokens_per_step,
                "mean_baseline_tokens_per_step": result.mean_baseline_tokens_per_step,
            }
        )

    print()
    summary = aggregate_results(rows)
    _write_csv(ROOT / "results_table.csv", rows)
    (ROOT / "summary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "sample_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    (ROOT / "results_table_latex.tex").write_text(_latex_table(summary), encoding="utf-8")
    print(f"Wrote {len(rows)} rows and aggregate artifacts to {ROOT}")


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
        "baseline_tokens",
        "mean_tokens_per_step",
        "mean_baseline_tokens_per_step",
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
WaaT transition accuracy & {summary["transition_accuracy_pct"]:.2f}\% \\
Prompt baseline transition accuracy & {summary["prompt_baseline_transition_accuracy_pct"]:.2f}\% \\
WaaT path accuracy & {summary["path_accuracy_pct"]:.2f}\% \\
Prompt baseline path accuracy & {summary["prompt_baseline_path_accuracy_pct"]:.2f}\% \\
WaaT terminal state accuracy & {summary["terminal_state_accuracy_pct"]:.2f}\% \\
Prompt baseline terminal accuracy & {summary["prompt_baseline_terminal_state_accuracy_pct"]:.2f}\% \\
Mean reasoning quality & {summary["mean_reasoning_quality"]:.2f} / 5 \\
Mean WaaT tokens / step & {summary["mean_tokens_per_step_waat"]:.2f} \\
Mean full-workflow tokens / step & {summary["mean_tokens_per_step_full_workflow_baseline"]:.2f} \\
Mean prompt baseline tokens / case & {summary["mean_prompt_baseline_tokens_per_case"]:.2f} \\
Token reduction & {summary["token_reduction_pct"]:.2f}\% \\
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
