# LangSmith regression evals

This directory contains reusable LangSmith evaluation helpers for DeepClaw.

## Files
- `langsmith_regression.py` — run baseline vs current-code experiments against an existing LangSmith dataset

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

## What the script evaluates
The current regression harness uses two metrics:
- `tool_call_required` — whether the run made a tool call when the dataset example requires one
- `retried_after_no_tool` — whether the run needed a retry/nudge after a no-tool first pass

## Notes
- The dataset examples are the evaluation contract; this harness intentionally assumes the dataset already exists in LangSmith.
- The script intentionally supports older baseline checkouts that may not have newer gateway helper functions.
- Results are written to a local JSON file so the repo stays clean.
