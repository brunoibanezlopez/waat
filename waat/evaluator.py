"""Evaluation metrics and reasoning-quality scoring."""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from typing import Any

from .agent import MODEL_NAME


class ReasoningEvaluator:
    """Scores update_workflow reasoning with Anthropic or a deterministic fallback."""

    def __init__(self, use_anthropic: bool | None = None, seed: int = 13) -> None:
        self.rng = random.Random(seed)
        self.use_anthropic = (
            bool(os.environ.get("ANTHROPIC_API_KEY")) if use_anthropic is None else use_anthropic
        )
        self.client = None
        if self.use_anthropic:
            try:
                from anthropic import Anthropic

                self.client = Anthropic()
            except Exception:
                self.use_anthropic = False

    def score_trace(self, trace: list[dict[str, Any]]) -> list[int]:
        return [self.score_reasoning(step["reasoning"], step) for step in trace]

    def score_reasoning(self, reasoning: str, step: dict[str, Any]) -> int:
        if self.use_anthropic and self.client is not None:
            try:
                return self._score_with_anthropic(reasoning, step)
            except Exception:
                pass
        return self._score_deterministic(reasoning, step)

    def _score_with_anthropic(self, reasoning: str, step: dict[str, Any]) -> int:
        prompt = {
            "rubric": {
                "1": "trivial, empty, or not grounded",
                "3": "adequate but generic",
                "5": "specific and well-grounded in workflow state and action result",
            },
            "reasoning": reasoning,
            "workflow_step": {
                "state": step["state"],
                "action_result": step["action_result"],
                "next_state": step["next_state"],
            },
            "instruction": "Return only a JSON object like {\"score\": 1}.",
        }
        response = self._anthropic_call_with_backoff(
            model=MODEL_NAME,
            max_tokens=80,
            messages=[{"role": "user", "content": json.dumps(prompt)}],
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        match = re.search(r"[1-5]", text)
        if not match:
            raise RuntimeError(f"No score found in evaluator response: {text!r}")
        return int(match.group(0))

    def _anthropic_call_with_backoff(self, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                return self.client.messages.create(**kwargs)
            except Exception as exc:
                last_error = exc
                time.sleep((2**attempt) + self.rng.random())
        raise RuntimeError("Anthropic evaluator call failed after retries") from last_error

    def _score_deterministic(self, reasoning: str, step: dict[str, Any]) -> int:
        words = reasoning.split()
        if len(words) < 20:
            return 1
        grounded_terms = [
            step["state"],
            step["next_state"],
            step["action_result"].get("agent", ""),
            "transition",
            "workflow",
            "structured",
        ]
        grounding_hits = sum(1 for term in grounded_terms if term and term in reasoning)
        if len(words) >= 35 and grounding_hits >= 4:
            return 5
        if grounding_hits >= 2:
            return 4
        return 3


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [score for row in rows for score in row["reasoning_scores"]]
    distribution = Counter(scores)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
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
        "mean_tokens_per_step_waat": round(sum(row["mean_tokens_per_step"] for row in rows) / len(rows), 2),
        "mean_tokens_per_step_full_workflow_baseline": round(
            sum(row["mean_baseline_tokens_per_step"] for row in rows) / len(rows), 2
        ),
        "mean_prompt_baseline_tokens_per_case": round(
            sum(row["prompt_baseline_tokens"] for row in rows) / len(rows), 2
        ),
        "token_reduction_pct": round(
            100
            * (
                1
                - (
                    sum(row["mean_tokens_per_step"] for row in rows)
                    / sum(row["mean_baseline_tokens_per_step"] for row in rows)
                )
            ),
            2,
        ),
    }
