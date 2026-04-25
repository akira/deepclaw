# LangSmith regression evals

This directory contains reusable LangSmith evaluation helpers for DeepClaw.

## Files
- `langsmith_regression.py` — run baseline vs current-code experiments against an existing LangSmith dataset
- `worker_run_case.py` — execute a single eval case against a specific repo checkout and print JSON for the harness

## Typical workflow

### 1. Curate the dataset in LangSmith
Use the `deepclaw` dataset as the source of truth. Add or edit examples there directly (or with one-off scripts) instead of keeping a checked-in trace seed file.

### 2. Run baseline vs current comparison
```bash
/home/ubuntu/deepclaw/.venv/bin/python evals/langsmith_regression.py \
  --dataset deepclaw \
  --repo /home/ubuntu/deepclaw \
  --baseline-commit origin/main \
  --baseline-worktree /tmp/deepclaw-eval-baseline \
  --results-path /tmp/deepclaw-evals/results.json
```

## Dataset contract
The dataset examples are the evaluation contract. At minimum, examples should include:
- `requires_tool_call`
- `label`

Recommended fields for stronger regression coverage:
- `expected_tool_names`
- `must_succeed_first_pass`
- `category`

## What the script evaluates
The current regression harness uses four metrics:
- `tool_call_required` — whether the run made a tool call when the dataset example requires one
- `expected_tool_names` — whether the run used the expected tool(s) when specified by the dataset example
- `first_pass_tool_use` — whether the run succeeded on the first pass when the dataset requires first-pass success
- `secondary_tool_recovery` — whether the gateway's retry/nudge path recovered tool use after a missed first pass, when first-pass success was required

## Notes
- The dataset examples are the evaluation contract; this harness intentionally assumes the dataset already exists in LangSmith.
- The script intentionally supports older baseline checkouts that may not have newer gateway helper functions.
- Results are written to a local JSON file so the repo stays clean.
