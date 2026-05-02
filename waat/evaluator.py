"""Evaluation metrics and reasoning-quality scoring."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .bedrock_client import BedrockClaudeClient


class ReasoningEvaluator:
    """Scores update_workflow reasoning with Claude on Bedrock."""

    def __init__(
        self,
        bedrock_profile: str | None = None,
        bedrock_region: str | None = None,
        bedrock_model_id: str | None = None,
        bedrock_verify_ssl: bool | str = True,
    ) -> None:
        self.bedrock_client = BedrockClaudeClient(
            profile_name=bedrock_profile,
            region_name=bedrock_region,
            model_id=bedrock_model_id,
            verify_ssl=bedrock_verify_ssl,
        )

    def score_trace(self, trace: list[dict[str, Any]]) -> list[int]:
        return [self.score_reasoning(step["reasoning"], step) for step in trace]

    def score_reasoning(self, reasoning: str, step: dict[str, Any]) -> int:
        system_prompt = (
            "You score workflow transition reasoning. Return only JSON like {\"score\": 5}. "
            "Use this rubric: 1 trivial or not grounded; 3 adequate but generic; "
            "5 specific and well-grounded in workflow state and action result."
        )
        payload = {
            "reasoning": reasoning,
            "workflow_step": {
                "state": step["state"],
                "action_result": step["action_result"],
                "next_state": step["next_state"],
            },
        }
        parsed, _response = self.bedrock_client.converse_json(system_prompt, payload, max_tokens=80)
        score = int(parsed.get("score", 0))
        if score < 1 or score > 5:
            raise ValueError(f"Invalid Bedrock evaluator score: {score}")
        return score


def aggregate_results(rows: list[dict[str, Any]], workflow_nodes: int | None = None) -> dict[str, Any]:
    scores = [score for row in rows for score in row["reasoning_scores"]]
    distribution = Counter(scores)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "num_cases": len(rows),
        "transition_accuracy_pct": round(100 * sum(row["transition_accuracy"] for row in rows) / len(rows), 2),
        "path_accuracy_pct": round(100 * sum(1 for row in rows if row["path_match"]) / len(rows), 2),
        "terminal_state_accuracy_pct": round(100 * sum(1 for row in rows if row["terminal_match"]) / len(rows), 2),
        "prompt_baseline_transition_accuracy_pct": round(
            100 * sum(row["prompt_baseline_transition_accuracy"] for row in rows) / len(rows), 2
        ),
        "prompt_baseline_path_accuracy_pct": round(
            100 * sum(1 for row in rows if row["prompt_baseline_path_match"]) / len(rows), 2
        ),
        "prompt_baseline_terminal_state_accuracy_pct": round(
            100 * sum(1 for row in rows if row["prompt_baseline_terminal_match"]) / len(rows), 2
        ),
        "mean_reasoning_quality": round(mean_score, 3),
        "reasoning_score_distribution": {str(score): distribution.get(score, 0) for score in range(1, 6)},
        "mean_tokens_per_case_waat": round(sum(row["total_tokens"] for row in rows) / len(rows), 2),
        "mean_tokens_per_step_waat": round(sum(row["mean_tokens_per_step"] for row in rows) / len(rows), 2),
        "mean_prompt_baseline_tokens_per_case": round(
            sum(row["prompt_baseline_tokens"] for row in rows) / len(rows), 2
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
    if workflow_nodes is not None:
        summary = {"workflow_nodes": workflow_nodes, **summary}
    return summary
