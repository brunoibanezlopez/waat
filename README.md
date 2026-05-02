# Workflow as a Tool (WaaT) Claude Sonnet Evaluation

This folder contains the evaluation prototype for the paper **Workflow as a Tool
(WaaT)**. It demonstrates how workflow state can be exposed as executable tools
instead of being embedded entirely in a prompt.

![Workflow Diagram](figures/workflow_diagram.png)

## What This Prototype Shows

- `check_workflow` and `update_workflow` enforce current-state grounding, valid
  state transitions, persisted reasoning, and pause/resume behavior.
- Claude Sonnet on Amazon Bedrock selects WaaT transitions from the full
  workflow definition, the current state specification, and production-shaped
  mock utility-agent output.
- Utility agents return normalized payloads with `status`, `data`,
  `external_refs`, `errors`, and `recommended_next_state`.
- A Claude Sonnet workflow-as-prompt baseline also receives the full YAML
  workflow as instruction context, but chooses transitions step by step without
  WaaT tools or runtime validation.
- A multi-turn workflow can pause for user confirmation and resume from the same
  workflow state after the user's reply.
- Claude Sonnet scores the transition reasoning quality.
- The benchmark includes adversarial synthetic cases so results are not
  oracle-perfect.

## Requirements

Install dependencies from this folder:

```bash
python -m pip install -r requirements.txt
```

The evaluation uses Amazon Bedrock through `boto3`. It expects AWS credentials
with `bedrock:InvokeModel` access to the configured inference profile.

## Evaluations

### Single-Turn Workflow Correctness

```bash
python run_evaluation.py
```

The runner loads simple AWS credentials from either `../.env` or `.env` before
creating the Bedrock client. The default model ID is the Vocus application
inference profile:

```bash
arn:aws:bedrock:ap-southeast-2:041538338020:application-inference-profile/umjk7k37bjmb
```

Override it with `--model-id` or `BEDROCK_MODEL_ID` if needed. Use `--limit N`
for a smoke test.

### Multi-Turn Confirmation

To run the Bedrock-backed multi-turn confirmation evaluation:

```bash
python run_multiturn_evaluation.py
```

### YAML Validation

To validate workflow YAML before interpreting benchmark results:

```bash
python run_yaml_validation.py
```

The validator performs local structural checks, then uses Claude Sonnet on
Bedrock to evaluate each branching state's transition conditions for overlap,
gap, and vagueness.

If your corporate network intercepts TLS, configure the corporate root CA as a
PEM bundle with either `AWS_CA_BUNDLE` or `--ca-bundle`. For a short local smoke
test only, you can pass `--no-verify-ssl`, but do not use that for paper
results.

## Outputs

- `results/single_turn/<nodes>_nodes/results_table.csv`: one row per synthetic test case.
- `results/single_turn/<nodes>_nodes/summary_stats.json`: aggregate
  single-turn metrics, including WaaT-vs-baseline token delta.
- `results/single_turn/<nodes>_nodes/sample_traces.json`: representative account, service, and complaint traces.
- `results/single_turn/<nodes>_nodes/results_table_latex.tex`: LaTeX table for the single-turn benchmark.
- `results/multiturn/confirmation/results_table.csv`: one row per multi-turn case.
- `results/multiturn/confirmation/summary_stats.json`: aggregate multi-turn metrics.
- `results/multiturn/confirmation/sample_traces.json`: first-turn and resumed traces.
- `results/validation/<workflow>/validation_report.json`: structural and semantic YAML validation report.
- `results/validation/<workflow>/semantic_findings.csv`: per-state overlap, gap, and vagueness findings.

## Structure

- `waat/workflow.py`: YAML loader and state graph.
- `waat/tools.py`: implementations of `check_workflow`, `update_workflow`, and session state.
- `waat/agent.py`: resumable WaaT evaluation loop using Claude Sonnet on Bedrock.
- `waat/contracts.py`: shared result contracts for utility agents and interaction states.
- `waat/baseline.py`: stepwise Claude Sonnet workflow-as-prompt baseline without WaaT tools.
- `waat/mock_agents.py`: deterministic mock utility agents with normalized payloads.
- `waat/evaluator.py`: Claude Sonnet reasoning-quality scorer.
- `waat/synthetic_data.py`: 30 synthetic customer requests.
- `waat/workflows/service_request.yaml`: workflow definition used by the evaluation.
- `waat/workflows/service_request_multiturn.yaml`: workflow with a user-confirmation pause.
- `run_yaml_validation.py`: Bedrock-backed workflow YAML validator for overlap, gap, and vagueness.

## Production Context

The prototype is intentionally small, but the pattern is motivated by production
customer-service agents where workflows are larger and more numerous. In
production, a request-level orchestrator may route simple requests directly to
utility agents and reserve WaaT for procedural workflows that require enforced
state, transition validation, auditability, and pause/resume behavior.
