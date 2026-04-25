---
name: langsmith-trace-regression-evals
description: Find new LangSmith regression traces, add them to the dataset, and run DeepClaw baseline-vs-current evals from the repo harness.
version: 1.0.0
---

# LangSmith Trace Regression Evals

Use this when you want to keep the `deepclaw` regression dataset fresh with real failures and then run the repo eval harness against that dataset.

## When to Use

- You want to mine real DeepClaw failures from LangSmith instead of inventing synthetic prompts
- You want to add new regression examples to the `deepclaw` dataset
- You want to compare `origin/main` vs current code with the same dataset
- You are debugging tool-use regressions, false completion, or no-tool narration

## Deterministic First

- Inspect actual LangSmith traces before writing extraction or evaluator logic
- Query root runs and run counts directly; do not infer from screenshots or memory
- Treat the LangSmith dataset as the eval contract, not ad hoc Python constants
- Use the DeepClaw repo venv, actual repo checkout, and a refreshed baseline worktree
- Keep `list_runs(..., limit=100)` unless intentionally paging

## Environment and Paths

- Repo: `/home/ubuntu/deepclaw`
- Venv Python: `/home/ubuntu/deepclaw/.venv/bin/python`
- Env file: `~/.deepclaw/.env`
- Dataset: `deepclaw`
- Harness: `evals/langsmith_regression.py`
- Default baseline worktree: `/tmp/deepclaw-eval-baseline`
- Default local results path: `/tmp/deepclaw-evals/results.json`

Load env for ad hoc scripts:

```python
from dotenv import load_dotenv
load_dotenv('/home/ubuntu/.deepclaw/.env', override=True)
```

## Part 1: Find Good New Traces

### What to Look For

Prefer reusable failure modes such as:
- `tool_runs == 0` when a tool clearly should have been called
- promise-only narration like `I can do that`, `I'll handle that`, or `If you want, I can...`
- false completion like `Done`, `Updated`, or `Fixed` with no real tool activity
- status answers that should have checked git, CI, filesystem, or runtime state first
- memory/preference acknowledgements with no persistence tool call

### Trace Triage Workflow

1. Identify the correct LangSmith project
   - usually `default` for DeepClaw unless env says otherwise
2. List recent **root** traces
3. Open candidate traces
4. Check:
   - first user message
   - tool run count
   - tool names
   - final assistant text
5. Add only traces that represent reusable regression cases

### Labeling Guidance

Use short, stable labels such as:
- `push-pr-no-tool`
- `investigate-gh-actions-no-tool`
- `remember-preference-direct`
- `false-completion-skill-update`
- `confirm-pr-pushed-without-verification`

A trace is worth adding if:
- the prompt is general enough to recur
- the failure mode is one we actively care about
- the expected behavior is clear
- it is not just a one-off auth or outage issue

## Part 2: Add New Traces to the Dataset

### Dataset Shape

Inputs:

```json
{"user_text": "Can you push the pr"}
```

Outputs:

```json
{
  "requires_tool_call": true,
  "label": "push-pr-no-tool",
  "expected_tool_names": ["execute"],
  "must_succeed_first_pass": true,
  "category": "git"
}
```

Metadata:

```json
{
  "source_trace_id": "019dc35a-f65b-7113-91e5-50d818b28167",
  "source_project": "default",
  "notes": "Promise-only reply with no git/tool inspection",
  "observed_tool_runs": 0,
  "observed_tool_names": [],
  "observed_llm_runs": 1,
  "observed_run_count": 1,
  "reference_assistant_text": "I can push that PR for you..."
}
```

### Rules

- dedupe by `source_trace_id`
- keep the dataset schema compact
- store provenance and stats in metadata
- keep `reference_assistant_text` trimmed
- use `limit=100` on `list_runs`

### One-Off Script Pattern

Run from the repo venv:

```bash
cd /home/ubuntu/deepclaw
/home/ubuntu/deepclaw/.venv/bin/python - <<'PY'
from dotenv import load_dotenv
from langsmith import Client, schemas

load_dotenv('/home/ubuntu/.deepclaw/.env', override=True)
client = Client()

dataset_name = 'deepclaw'
project_name = 'default'
trace_specs = [
    {
        'trace_id': 'TRACE_ID_HERE',
        'label': 'push-pr-no-tool',
        'requires_tool_call': True,
        'notes': 'Short explanation of the failure mode',
    },
]


def ensure_dataset(name: str):
    datasets = list(client.list_datasets(dataset_name=name, limit=5))
    if datasets:
        return datasets[0]
    return client.create_dataset(
        name,
        description='DeepClaw regression dataset sourced from real LangSmith traces.',
        data_type=schemas.DataType.kv,
        metadata={'project': 'deepclaw', 'purpose': 'trace-sourced-regression-evals'},
    )


def get_root_run(trace_id: str):
    roots = list(client.list_runs(project_name=project_name, trace_id=trace_id, is_root=True, limit=5))
    if not roots:
        raise RuntimeError(f'No root run found for trace {trace_id}')
    return roots[0]


def get_trace_stats(trace_id: str):
    runs = list(client.list_runs(project_name=project_name, trace_id=trace_id, limit=100))
    tool_runs = [run for run in runs if run.run_type == 'tool']
    llm_runs = [run for run in runs if run.run_type == 'llm']
    return {
        'observed_tool_runs': len(tool_runs),
        'observed_tool_names': sorted({run.name for run in tool_runs}),
        'observed_llm_runs': len(llm_runs),
        'observed_run_count': len(runs),
    }


def latest_ai_text(root_run):
    outputs = root_run.outputs or {}
    messages = outputs.get('messages') or []
    for msg in reversed(messages):
        if msg.get('type') != 'ai':
            continue
        content = msg.get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    parts.append(block.get('text', ''))
            return '\n'.join(part for part in parts if part)
    return ''


def list_existing_trace_ids(dataset_id: str):
    seen = set()
    offset = 0
    page_size = 100
    while True:
        examples = list(client.list_examples(dataset_id=dataset_id, offset=offset, limit=page_size))
        if not examples:
            break
        for example in examples:
            trace_id = (example.metadata or {}).get('source_trace_id')
            if trace_id:
                seen.add(trace_id)
        if len(examples) < page_size:
            break
        offset += page_size
    return seen


dataset = ensure_dataset(dataset_name)
existing = list_existing_trace_ids(str(dataset.id))
new_examples = []

for spec in trace_specs:
    if spec['trace_id'] in existing:
        print(f"skip existing {spec['trace_id']}")
        continue
    root_run = get_root_run(spec['trace_id'])
    messages = (root_run.inputs or {}).get('messages') or []
    if not messages:
        print(f"skip no input messages {spec['trace_id']}")
        continue
    user_text = messages[0].get('content')
    if not user_text:
        print(f"skip empty user text {spec['trace_id']}")
        continue
    stats = get_trace_stats(spec['trace_id'])
    new_examples.append({
        'inputs': {'user_text': user_text},
        'outputs': {
            'requires_tool_call': spec['requires_tool_call'],
            'label': spec['label'],
        },
        'source_run_id': str(root_run.id),
        'metadata': {
            'source_trace_id': spec['trace_id'],
            'source_project': project_name,
            'label': spec['label'],
            'notes': spec['notes'],
            'reference_assistant_text': latest_ai_text(root_run)[:1000],
            **stats,
        },
    })

if new_examples:
    client.create_examples(dataset_id=dataset.id, examples=new_examples)

print({
    'dataset': dataset_name,
    'dataset_id': str(dataset.id),
    'examples_added': len(new_examples),
})
PY
```

## Part 3: Run the Evals

Standard command:

```bash
cd /home/ubuntu/deepclaw
/home/ubuntu/deepclaw/.venv/bin/python evals/langsmith_regression.py \
  --dataset deepclaw \
  --repo /home/ubuntu/deepclaw \
  --baseline-commit origin/main \
  --baseline-worktree /tmp/deepclaw-eval-baseline \
  --results-path /tmp/deepclaw-evals/results.json
```

What the harness does:
- refreshes or creates the baseline worktree
- runs the dataset against the baseline checkout
- runs the same dataset against the current checkout
- writes a local JSON summary
- records LangSmith experiment URLs

### Current metrics
- `tool_call_required`
- `expected_tool_names`
- `first_pass_tool_use`
- `secondary_tool_recovery`

### Read the Results Correctly

Do not only report a single aggregate number. Call out:
- overall summary metrics
- which examples improved
- which examples still fail
- whether retry behavior improved or regressed separately from first-pass tool use

## DeepClaw-Specific Cautions

- Use the repo venv, not system Python
- Worker subprocesses must evaluate against the repo checkout, not an empty temp workspace
- Refresh the baseline worktree before reuse
- Old baselines may not have the newest gateway helpers; use tolerant imports and fallbacks in worker logic
- LangSmith eval result rows may be dicts or objects; handle both

## Verification

- Confirm the skill has valid YAML frontmatter and the required sections
- After adding traces, verify labels, `source_trace_id`, and `requires_tool_call` are correct
- After running the harness, save or report:
  - dataset name/id
  - baseline experiment URL
  - post-change experiment URL
  - local JSON results path
  - notable wins
  - remaining gaps
- Re-run the same dataset after prompt or runtime changes so comparisons stay apples-to-apples
