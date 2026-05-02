"""Run WaaT scaling experiments across 6, 20, 50, and 100-node workflows."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import yaml

from waat.workflow_variants import TARGET_NODE_COUNTS, write_variants, workflow_node_count


ROOT = Path(__file__).resolve().parent
BASE_WORKFLOW_PATH = ROOT / "waat" / "workflows" / "service_request.yaml"
VARIANT_DIR = ROOT / "waat" / "workflows" / "generated"
RUNS_DIR = ROOT / "results" / "runs"
SCALING_DIR = ROOT / "results" / "scaling"
FIGURES_DIR = ROOT / "figures"


def main() -> None:
    args = _parse_args()
    variant_paths = write_variants(BASE_WORKFLOW_PATH, VARIANT_DIR, TARGET_NODE_COUNTS)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.scaling_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for path in variant_paths:
        node_count = _node_count(path)
        print(f"\n=== Evaluating {node_count}-node workflow ===", flush=True)
        command = [
            sys.executable,
            "run_evaluation.py",
            "--workflow-path",
            str(path),
            "--output-prefix",
            f"aws_{node_count}",
            "--output-dir",
            str(args.runs_dir),
            "--region",
            args.region,
            "--model-id",
            args.model_id,
        ]
        if args.profile:
            command.extend(["--profile", args.profile])
        if args.ca_bundle:
            command.extend(["--ca-bundle", args.ca_bundle])
        if args.no_verify_ssl:
            command.append("--no-verify-ssl")
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        subprocess.run(command, cwd=ROOT, check=True)
        summaries.append(json.loads((args.runs_dir / f"aws_{node_count}_summary_stats.json").read_text(encoding="utf-8")))

    _write_scaling_summary(summaries, args.scaling_dir)
    _render_scaling_plot(summaries, args.figures_dir)
    for path in variant_paths:
        _render_workflow_graph(path, args.figures_dir)

    print("\nWrote scaling summaries and figures.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaling evaluation for generated WaaT workflows.")
    parser.add_argument("--profile", help="AWS profile name, for example an SSO profile.")
    parser.add_argument("--region", default="ap-southeast-2", help="AWS Bedrock runtime region.")
    parser.add_argument(
        "--model-id",
        default="arn:aws:bedrock:ap-southeast-2:041538338020:application-inference-profile/umjk7k37bjmb",
        help="Amazon Bedrock model ID, inference profile ID, or application inference profile ARN.",
    )
    parser.add_argument("--ca-bundle", help="Path to a corporate CA bundle PEM file for AWS SSL verification.")
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable AWS SSL verification for smoke tests.")
    parser.add_argument("--limit", type=int, help="Run only the first N synthetic cases per workflow size.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Directory for per-size run artifacts.")
    parser.add_argument("--scaling-dir", type=Path, default=SCALING_DIR, help="Directory for scaling summary artifacts.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Directory for generated PNG figures.")
    return parser.parse_args()


def _node_count(path: Path) -> int:
    return workflow_node_count(yaml.safe_load(path.read_text(encoding="utf-8")))


def _write_scaling_summary(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    fields = [
        "workflow_nodes",
        "num_cases",
        "transition_accuracy_pct",
        "path_accuracy_pct",
        "terminal_state_accuracy_pct",
        "prompt_baseline_transition_accuracy_pct",
        "prompt_baseline_path_accuracy_pct",
        "prompt_baseline_terminal_state_accuracy_pct",
        "mean_reasoning_quality",
        "mean_tokens_per_case_waat",
        "mean_tokens_per_step_waat",
        "mean_prompt_baseline_tokens_per_case",
        "case_token_reduction_pct",
    ]
    with (output_dir / "scaling_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})
    (output_dir / "scaling_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def _render_scaling_plot(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    summaries = sorted(summaries, key=lambda item: item["workflow_nodes"])
    nodes = [summary["workflow_nodes"] for summary in summaries]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(nodes, [s["mean_tokens_per_case_waat"] for s in summaries], marker="o", label="WaaT")
    axes[0].plot(
        nodes,
        [s["mean_prompt_baseline_tokens_per_case"] for s in summaries],
        marker="s",
        label="Prompt baseline",
    )
    axes[0].set_title("Token Usage")
    axes[0].set_xlabel("Workflow nodes")
    axes[0].set_ylabel("Mean tokens per case")
    axes[0].legend()

    axes[1].plot(nodes, [s["terminal_state_accuracy_pct"] for s in summaries], marker="o", label="WaaT terminal")
    axes[1].plot(
        nodes,
        [s["prompt_baseline_terminal_state_accuracy_pct"] for s in summaries],
        marker="s",
        label="Baseline terminal",
    )
    axes[1].plot(nodes, [s["path_accuracy_pct"] for s in summaries], marker="o", linestyle="--", label="WaaT path")
    axes[1].plot(
        nodes,
        [s["prompt_baseline_path_accuracy_pct"] for s in summaries],
        marker="s",
        linestyle="--",
        label="Baseline path",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Workflow nodes")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "scaling_metrics.png", dpi=200)
    plt.close(fig)


def _render_workflow_graph(path: Path, output_dir: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    for state_id, state in data["states"].items():
        graph.add_node(state_id)
        for transition in state.get("transitions", []):
            graph.add_edge(state_id, transition["to"])
    for terminal_state in data["terminal_states"]:
        graph.add_node(terminal_state)

    positions = _layout_graph(graph)
    node_count = workflow_node_count(data)
    colors = [
        "#6fb6ff" if node not in data["terminal_states"] else ("#72d572" if node == "REQUEST_COMPLETE" else "#ff7b7b")
        for node in graph.nodes
    ]
    fig_width = 10 if node_count <= 20 else 14
    fig_height = 7 if node_count <= 20 else 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    nx.draw_networkx_edges(graph, positions, ax=ax, arrows=True, alpha=0.35, width=0.8, arrowsize=8)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=240 if node_count <= 20 else 80)
    if node_count <= 20:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=7)
    else:
        labels = {
            node: node
            for node in graph.nodes
            if node in {
                "CLASSIFY_REQUEST",
                "VERIFY_CUSTOMER_IDENTITY",
                "RETRIEVE_CUSTOMER_PROFILE",
                "HANDLE_ACCOUNT_QUERY",
                "HANDLE_SERVICE_CHANGE",
                "HANDLE_COMPLAINT",
                "CHECK_CONTRACT_TERMS",
                "CHECK_SERVICE_OUTAGE",
                "CREATE_SUPPORT_TICKET",
                "REQUEST_COMPLETE",
                "REQUEST_ESCALATED",
            }
        }
        nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=7)
    ax.set_title(f"{node_count}-Node SERVICE_REQUEST Workflow")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"workflow_graph_{node_count}_nodes.png", dpi=200)
    plt.close(fig)


def _layout_graph(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    if len(graph.nodes) > 20:
        return nx.spring_layout(graph, seed=7, k=1.2, iterations=150)
    return nx.spring_layout(graph, seed=7, k=1.6, iterations=200)


if __name__ == "__main__":
    main()
