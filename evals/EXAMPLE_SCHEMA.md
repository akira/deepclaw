# DeepClaw Eval Example Schema

The LangSmith `deepclaw` dataset is the eval contract.

## Required fields

Every example should include these output fields:
- `label` — stable short identifier like `remember-preference`
- `requires_tool_call` — whether a tool call is required for success
- `category` — high-level grouping like `memory-and-skills` or `conversation`

## Strongly recommended fields

Use these whenever they are well-defined:
- `expected_tool_names` — list of acceptable tools for the task
- `must_succeed_first_pass` — whether the first assistant pass must already use the correct tool behavior
- `eval_mode` — `compliance` or `live`
- `behavior_tags` — list of narrower behavior labels such as `memory_persistence`, `no_tool_conversation`, `read_only_inspection`
- `side_effect_risk` — `low`, `medium`, or `high`

## Phase-2 / future fields

These fields support ideal-trajectory and efficiency scoring on simple tasks. They are optional today and should only be populated when the expectations are clear.
- `expected_first_tool`
- `ideal_tool_count`
- `ideal_ai_turn_count`
- `max_runtime_class`

## Metadata provenance

When an example comes from a real trace, include these metadata fields:
- `source_trace_id`
- `notes`
- `reference_assistant_text`

## Example

```json
{
  "inputs": {
    "user_text": "Can you remember that when I ask you to push the PR use the DeepClaw dev skill and also open the PR?"
  },
  "outputs": {
    "label": "remember-preference",
    "requires_tool_call": true,
    "category": "memory-and-skills",
    "expected_tool_names": ["edit_file"],
    "must_succeed_first_pass": true,
    "eval_mode": "compliance",
    "behavior_tags": ["memory_persistence", "user_preference"],
    "side_effect_risk": "low"
  },
  "metadata": {
    "source_trace_id": "019dc35a-f65b-7113-91e5-50d818b28167",
    "notes": "Promise-only reply with no persistence tool call",
    "reference_assistant_text": "I can do that."
  }
}
```
