---
name: skillify-pattern
description: Turn a successful workflow or recurring failure into a durable DeepClaw skill with deterministic code, tests, audits, and verification.
version: 1.0.0
---

# Skillify Pattern

Use this when a fix, workaround, or repeated workflow should become permanent infrastructure instead of session-only knowledge.

## When to Use
- A bug or failure happened more than once
- A one-off workflow finally worked and should be reusable next time
- The user says "skillify it", "remember this as a skill", or asks to make a pattern permanent
- A task currently relies on model memory where deterministic code or a repeatable procedure would be safer

## Deterministic First
- If part of the workflow can be expressed as deterministic code, scripts, or direct tool checks, move it out of free-form reasoning.
- Prefer code for precision work: path checks, time math, URL reachability, parsing, diffing, filtering, and state lookups.
- The model may design the workflow, but future runs should be constrained by deterministic helpers whenever possible.

## Workflow
1. Name the repeated problem or successful pattern clearly.
2. Define the trigger: what user request or failure mode should route here next time?
3. Write or update `SKILL.md` with exact usage criteria and the required workflow.
4. Add deterministic code or scripts when the task includes precision work the model should not improvise.
5. Add tests for the deterministic pieces and for the routing/invocation path if applicable.
6. Add verification steps so future sessions can confirm the skill actually worked.
7. Audit for overlap with existing skills; avoid duplicate triggers and duplicate descriptions.
8. If the skill writes to long-term storage, define where outputs belong.

## Verification
- Confirm the resulting skill has a clear `When to Use` section.
- Confirm any deterministic helper has a test or direct verification command.
- Confirm the skill can be discovered and loaded by the local skills system.
- Confirm the skill does not overlap ambiguously with an existing skill.
