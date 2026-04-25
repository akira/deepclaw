# LangSmith regression evals

This directory contains reusable LangSmith evaluation helpers for DeepClaw.

## Files
- `langsmith_regression.py` — populate a LangSmith dataset from trace seeds, then run baseline vs current-code experiments
- `datasets/deepclaw_seed_traces.json` — bootstrap trace list for the `deepclaw` regression dataset

## Typical workflow

### 1. Populate the dataset from trace seeds
```bash
/home/ubuntu/deepclaw/.venv/bin/python evals/langsmith_regression.py \
  --project default \
  --dataset deepclaw \
  --seed-traces-path evals/datasets/deepclaw_seed_traces.json \
  --populate-only
```

### 2. Run baseline vs current comparison
```bash
/home/ubuntu/deepclaw/.venv/bin/python evals/langsmith_regression.py \
  --project default \
  --dataset deepclaw \
  --repo /home/ubuntu/deepclaw \
  --baseline-commit origin/main \
  --baseline-worktree /tmp/deepclaw-eval-baseline \
  --results-path /tmp/deepclaw-evals/results.json
```

## What the script evaluates
The current regression harness uses two metrics:
- `tool_call_required` — whether the run made a tool call when the dataset example requires one
- `retried_after_no_tool` — whether the run needed a retry/nudge after a no-tool first pass

## Notes
- The dataset examples are the evaluation contract; the seed JSON is only a bootstrap source for populating the dataset.
- The script intentionally supports older baseline checkouts that may not have newer gateway helper functions.
- Results are written to a local JSON file so the repo stays clean.
