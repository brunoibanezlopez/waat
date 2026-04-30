# Workflow as a Tool (WaaT) Strands Prototype

This folder contains a shareable Strands-based research prototype for the paper
**Workflow as a Tool (WaaT)**. It demonstrates how workflow state can be exposed
as executable tools instead of being embedded entirely in a prompt.

![Workflow Diagram](workflow_diagram.png)

## What This Prototype Shows

- `check_workflow` and `update_workflow` are implemented as Strands `@tool`
  functions.
- A Strands `Agent` is constructed with the WaaT tools attached, matching the
  production integration style.
- The offline evaluation calls the Strands-decorated tools directly so results
  are deterministic and reproducible without live model variance.
- A pure workflow-as-prompt baseline is included for comparison.
- The benchmark includes adversarial synthetic cases so results are not
  oracle-perfect.

## Requirements

Install dependencies from this folder:

```bash
python -m pip install -r requirements.txt
```

The core dependency is `strands-agents`. Anthropic support is included for
experimentation, but the default evaluation does not require API credentials.

## Run

```bash
python run_evaluation.py
```

The default evaluation uses deterministic mock utility agents and deterministic
reasoning scoring so the paper artifacts are reproducible without API
credentials. `WaaTSuperAgent` and `ReasoningEvaluator` also include Anthropic SDK
integration points for `claude-sonnet-4-20250514` and exponential backoff.

## Outputs

- `results_table.csv`: one row per synthetic test case.
- `summary_stats.json`: aggregate WaaT and prompt-baseline metrics.
- `sample_traces.json`: representative account, service, and complaint traces.
- `results_table_latex.tex`: IEEE-style LaTeX aggregate table.
- `results_section.tex`: results section for the paper.
- `claude_update_section.tex`: consolidated LaTeX section to give to Claude.

Current aggregate results:

- WaaT transition accuracy: `85.00%`
- Prompt baseline transition accuracy: `80.00%`
- WaaT path accuracy: `83.33%`
- Prompt baseline path accuracy: `73.33%`
- WaaT terminal accuracy: `93.33%`
- Prompt baseline terminal accuracy: `80.00%`
- Step-level token reduction versus full-workflow prompting: `58.71%`

## Structure

- `waat/workflow.py`: YAML loader and state graph.
- `waat/tools.py`: Strands `@tool` implementations of `check_workflow` and `update_workflow`.
- `waat/agent.py`: Strands super-agent construction plus reproducible evaluation loop.
- `waat/baseline.py`: pure workflow-as-prompt baseline without WaaT tools.
- `waat/mock_agents.py`: deterministic mock utility agents.
- `waat/evaluator.py`: reasoning-quality scorer.
- `waat/synthetic_data.py`: 30 synthetic customer requests.
- `waat/workflows/service_request.yaml`: workflow definition used by the evaluation.

## Production Context

The prototype is intentionally small, but the pattern is motivated by production
customer-service agents where workflows are larger and more numerous. In
production, a super-agent may coordinate assurance, billing, cancellations,
customer care, service inventory, ticket updates, and account administration.
WaaT keeps this procedural complexity out of the prompt and exposes only the
current workflow state through tools.
