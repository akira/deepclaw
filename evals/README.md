# LangSmith regression evals

This directory contains reusable LangSmith evaluation helpers for DeepClaw.

## Files
- `langsmith_regression.py` — run baseline vs current-code experiments against an existing LangSmith dataset
- `langsmith_pairwise.py` — run two source experiments and compare them head-to-head with a pairwise evaluator
- `worker_run_case.py` — execute a single eval case against a specific repo checkout and print JSON for the harness
- `dataset_audit.py` — inspect dataset annotation coverage, category spread, and missing provenance
- `EXAMPLE_SCHEMA.md` — the recommended dataset output/metadata contract

## Eval modes
DeepClaw currently treats every run as a **live agent execution**. Operationally, we distinguish between:
- **compliance** examples — cheaper behavior/routing checks that should eventually run in a first-tool / bounded mode
- **live** examples — end-to-end scenarios that are realistic but more expensive and riskier

The dataset stores this distinction in the `eval_mode` field. The current harness uses that field for filtering so you can run targeted subsets even before the runner gains a separate cheap execution engine.

## Typical workflow

### 1. Curate the dataset in LangSmith
Use the `deepclaw` dataset as the source of truth. Add or edit examples there directly (or with one-off scripts) instead of keeping a checked-in trace seed file.

### 1a. Audit the dataset contract
```bash
python -m evals.dataset_audit --dataset deepclaw
```

### 2. Run baseline vs current comparison
```bash
python -m evals.langsmith_regression \
  --dataset deepclaw \
  --repo "$(pwd)" \
  --baseline-commit origin/main \
  --baseline-worktree /tmp/deepclaw-eval-baseline \
  --category memory-and-skills \
  --eval-mode compliance \
  --results-path /tmp/deepclaw-evals/results.json
```

### 3. Run pairwise comparison between two models
```bash
python -m evals.langsmith_pairwise \
  --dataset deepclaw \
  --repo "$(pwd)" \
  --model-a anthropic:claude-haiku-4-5 \
  --model-b openai:gpt-4o-mini \
  --category codebase-inspection \
  --eval-mode compliance \
  --example-limit 5 \
  --results-path /tmp/deepclaw-evals/pairwise-results.json
```

For quick smoke tests, prefer `--example-limit 2` or a couple of `--example-id` flags before scaling up to the full dataset.

## Dataset contract
The dataset examples are the evaluation contract. At minimum, examples should include:
- `label`
- `requires_tool_call`
- `category`

Recommended fields for stronger regression coverage:
- `expected_tool_names`
- `must_succeed_first_pass`
- `eval_mode`
- `behavior_tags`
- `side_effect_risk`

See `evals/EXAMPLE_SCHEMA.md` for the fuller contract, provenance fields, and example payloads.

## Supported filters
Both runners support targeted filtering after loading examples from LangSmith:
- `--example-id <uuid>`
- `--example-limit N`
- `--category <name>`
- `--eval-mode compliance|live`
- `--behavior-tag <tag>`
- `--side-effect-risk low|medium|high`

These filters compose. For example, you can run only low-risk compliance examples in `memory-and-skills`.

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
- Pairwise evaluation compares **two existing source experiments**. The pairwise result is easiest to inspect via the LangSmith compare URL that includes both `selectedSessions` and `comparativeExperiment`.
- **Important:** these evals are currently **live agent runs**, not dry-run classifiers. They load real credentials, can call real tools, and can spend real money. Start with `--example-limit 2` or explicit `--example-id` values.
- If local LangSmith uploads fail because an inherited CA bundle path is invalid, prefer running with:
  - `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`
  - `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`
