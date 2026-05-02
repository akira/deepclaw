# DeepClaw Context Architecture — Post PR #30 Redesign

## Goal

Define a fresh context-engineering design for DeepClaw **after** PR #30 (`deepagents-context-engineering`).

This document replaces the earlier custom thread-compaction-first design as the primary recommendation.

The new goal is:

> Keep the wins from the DeepAgents CLI-style approach, then add the missing DeepClaw-owned state, routing, and continuity layers **around** that substrate instead of building a competing compaction stack first.

---

## What changed in PR #30

PR #30 landed a meaningful architectural shift:

- adopted **DeepAgents CLI-style local context detection**
- adopted **DeepAgents summarization middleware** via `create_summarization_tool_middleware(...)`
- introduced a **CompositeBackend** with artifact routes:
  - `/large_tool_results/`
  - `/conversation_history/`
- kept the existing stable layers:
  - `SOUL.md`
  - `MemoryMiddleware`
  - `SkillsMiddleware`
  - DeepAgents subagents

In practical terms, DeepClaw is no longer in the same pre-redesign state as before. It now has:

1. a better runtime substrate for offloading and summarization
2. a better local/project context signal
3. an existing framework-level compaction path

That changes what the next design should optimize for.

---

## Why the original design is no longer the right default

The original design centered on a DeepClaw-owned `compaction.py` orchestrator, explicit thread rotation, summary artifacts, and lineage metadata.

That was a coherent plan **before** PR #30.

After PR #30, the main problem is no longer “DeepClaw has no compaction/offload substrate.”
Now the problem is:

- DeepClaw has **runtime compaction plumbing**, but
- it still lacks a **state-first context model**,
- lacks a **policy layer** for route selection,
- lacks **structured continuity state** for long technical tasks,
- and still risks treating transcript residue as working memory.

So the fresh design should be:

> **DA substrate first, DeepClaw state and routing on top, explicit thread rotation only as a fallback or boundary tool — not as the primary architecture.**

---

## Current post-PR-30 architecture

### Stable layers already present

#### 1. Always-on prompt
- `SOUL.md`
- tool-use enforcement

#### 2. Durable memory
- `MemoryMiddleware`
- source: `~/.deepclaw/AGENTS.md`

#### 3. Durable skills
- `SkillsMiddleware`
- source: `~/.deepclaw/skills/`

#### 4. Local/project context
- `LocalContextMiddleware`
- detects cwd, git, runtimes, package managers, tests, top-level files
- injects a compact local context block into the system prompt
- refreshes after DeepAgents summarization cutoff events

#### 5. Framework-level transcript compaction/offload
- DeepAgents summarization middleware
- `CompositeBackend` routes:
  - `/large_tool_results/`
  - `/conversation_history/`

#### 6. Checkpointed thread persistence
- LangGraph checkpointer
- Telegram still maps `chat_id -> thread_id`

#### 7. Subagents
- still available via DeepAgents subagent support

---

## What is still missing

PR #30 improved the substrate, but it did **not** solve the full long-horizon context problem.

### 1. No structured working-state layer
DeepClaw still reasons largely from:
- prompt
- memory
- skills
- transcript tail/history
- local-context block

What is still missing is a compact task-state object such as:
- active goal
- current repo/branch/PR
- current subproblem
- next action
- relevant files
- blockers
- decisions made
- recent failures
- attempt log
- artifact refs

Without this, transcript history still does too much conceptual work.

### 2. No explicit context-routing policy
DeepAgents summarization exists, but DeepClaw does not yet have a first-class policy layer that decides:
- when to keep context as-is
- when to shrink artifacts only
- when to compact historical transcript
- when to rebuild from state
- when to rotate thread or hard-reset

### 3. No recoverability-oriented continuity contract
The system can offload and summarize, but it does not yet have a DeepClaw-owned continuity schema that guarantees preservation of:
- objective
- decisions
- unresolved issues
- files/artifacts
- recent failures
- next best action

### 4. No explicit decision log / attempt log
This is one of the biggest remaining gaps for long coding/debugging sessions.

DeepClaw needs to preserve:
- what has already been tried
- what failed
- what was ruled out
- what decision was made and why

This is more valuable than keeping more raw transcript.

### 5. Local context is static and broad, not relevance-aware
The current local-context block is useful, but still blunt:
- one general script
- broad filesystem listing
- no task-aware ranking
- no distinction between always-useful vs situational context

### 6. Thread identity is still too thin
Telegram still effectively uses `chat_id -> thread_id` state.
There is no richer notion of:
- session mode
- checkpoint lineage
- boundary reasons
- continuity refs
- compaction/rebuild metadata

This is no longer the first thing to solve, but it still matters.

---

## Fresh design principles

### 1. Keep the DeepAgents substrate
Do **not** replace the DeepAgents summarization/offload path with a parallel custom system as the first move.

Use it as the substrate.

### 2. Move toward a state-first prompt model
Transcript should be evidence, not the primary working memory.

Each turn should increasingly be assembled from:
1. stable control
2. structured working state
3. selected local/project context
4. active transcript tail
5. artifact previews/references
6. historical continuity summary only when needed

### 3. Separate runtime hygiene from continuity logic
PR #30 improved runtime hygiene.
The next phase should focus on continuity and task-state quality, not just further transcript cleanup.

### 4. Treat compaction as one tool in a routing ladder
Compaction is not the only move.
The system should first ask:
- is the problem artifacts?
- is the problem stale transcript?
- is the problem missing task-state representation?
- is the problem that the current thread should be rebuilt from state?

### 5. Optimize for long technical sessions
The target workload is:
- coding
- debugging
- trace inspection
- PR review
- deployment
- architecture iteration
- repo spelunking
- multi-hour sessions with pivots and resumptions

The system should optimize for:
- continuation quality
- execution sharpness
- recoverability
- decision retention
- anti-repetition

---

## Proposed architecture

## Layer A — Stable control
**Purpose:** invariant steering and durable identity.

**Contents:**
- `SOUL.md`
- tool-use enforcement
- durable memory (`AGENTS.md`)
- durable skills (`skills/`)

**Policy:**
- keep this layer compact and stable
- do not let project/task residue accrete here

---

## Layer B — Derived environment context
**Purpose:** compact project/runtime grounding.

**Contents:**
- cwd
- repo root
- git branch / dirty state
- language/runtime
- package manager
- test command hints
- relevant workspace shape

**Policy:**
- make this relevance-aware and compact
- task-sensitive selection should eventually replace the current one-size-fits-all block

This layer should answer:
> “What environment am I operating in right now?”

---

## Layer C — Structured working state
**Purpose:** DeepClaw’s primary live task memory.

This is the main new layer.

**Suggested schema:**

```yaml
session:
  mode: active
  objective: "Investigate post-PR-30 context design"
  status: in_progress

workspace:
  repo: /home/ubuntu/deepclaw
  branch: deepagents-context-engineering
  pr: 30

focus:
  current_subproblem: "design next context architecture"
  next_action: "write redesign doc and PR stack"

state:
  relevant_files:
    - deepclaw/agent.py
    - deepclaw/local_context.py
    - deepclaw/gateway.py
  decisions:
    - "keep DeepAgents summarization as substrate"
    - "add DeepClaw state/routing above it"
  blockers: []

attempts:
  tried:
    - "custom thread-compaction-first design"
  avoid_repeating:
    - "do not build a competing compaction stack first"

artifacts:
  refs: []
```

**Properties:**
- compact
- structured
- task-oriented
- updated incrementally
- checkpointed outside the prompt when possible

This layer should answer:
> “What task state matters right now?”

---

## Layer D — Active transcript tail
**Purpose:** preserve immediate conversational continuity.

**Contents:**
- recent unresolved user/assistant turns
- recent tool activity still relevant to the next step

**Policy:**
- tightly bounded
- unresolved only
- stale tool chatter should not dominate
- transcript tail supplements working state; it does not replace it

This layer should answer:
> “What just happened that still matters for the next move?”

---

## Layer E — Artifact/reference layer
**Purpose:** preserve bulky material without polluting the prompt.

**Contents:**
- large tool outputs
- large pasted user payloads
- logs
- diffs
- traces
- browser captures
- search dumps
- history snapshots

**Policy:**
- inline only previews, summaries, and refs
- raw payloads live in artifacts
- make retrieval explicit and recoverable

This layer should answer:
> “Where is the raw detail if we need it?”

---

## Layer F — Historical continuity state
**Purpose:** preserve continuity across long sessions, compaction, rebuilds, and thread boundaries.

Unlike the older design, this should not start as a custom per-thread artifact system.
Instead, it should be a **DeepClaw-owned continuity schema** that can be:
- checkpointed privately
- serialized for handoffs
- referenced during rebuilds
- optionally exported during explicit thread rotations

**Minimum fields:**
- objective
- status
- key decisions
- unresolved issues
- files/artifacts
- recent failures
- next best action
- anti-repetition notes

This layer should answer:
> “If we had to resume cleanly, what must survive?”

---

## Layer G — Runtime route selector
**Purpose:** choose the lightest successful context strategy before model invocation.

This is the missing policy layer.

Suggested routes:

### Route 1 — `fits`
Use assembled context as-is.

### Route 2 — `shrink_artifacts`
If the overage or clutter is mostly bulky artifact previews:
- shrink previews
- preserve working state and transcript tail

### Route 3 — `compact_history`
If history is the problem:
- let DeepAgents summarization compact older transcript
- preserve state and recent tail

### Route 4 — `rebuild_from_state`
If context is still too large or low-quality:
- rebuild from Layers A/B/C plus tiny transcript tail and artifact refs
- deprioritize most transcript residue

### Route 5 — `boundary_reset`
If the thread is pathological:
- checkpoint continuity state
- create a fresh thread boundary
- continue from state + refs

This is where the fresh design most differs from the old one:
- thread rotation becomes a **late fallback / explicit boundary tool**
- not the primary compaction architecture

---

## Role of thread rotation after PR #30

Thread rotation is still useful, but its role changes.

### Good uses
- explicit `/new`
- explicit `/clear`
- pathological overflow recovery
- major mode or task boundary
- model-switch boundary if needed
- manual operator cleanup

### Not the default first response
Do not use thread rotation as the first-line answer to ordinary transcript growth.

First use:
- artifact shrinking
- DeepAgents compaction
- rebuild-from-state

Only rotate when those are insufficient or when a clean boundary is semantically useful.

---

## What remains valid from the original design

The earlier design was still right about several things:

### Still valid
- do not treat one sticky thread as long-term memory
- keep active context bounded and high-signal
- separate context from durable memory
- offload bulky artifacts instead of reinjecting them inline
- preserve recoverability
- use subagents for context isolation
- summaries should be non-authoritative historical reference

### No longer the default implementation path
- a bespoke `deepclaw/compaction.py` as the main architecture
- thread-rotation-first design
- rich per-thread lineage metadata as the first priority
- `/compact` as the primary UX primitive

---

## Recommended next implementation direction

## Phase 1 — Make the DA substrate measurable and controllable
Add instrumentation for:
- prompt size estimates by layer
- artifact preview size
- transcript-tail size
- local-context size
- when DeepAgents summarization triggers
- whether rebuild-from-state would have been cheaper

Outcome:
- make context-routing decisions observable

---

## Phase 2 — Add structured working state
Introduce a private checkpointed working-state object.

This should include:
- objective
- focus
- next action
- decisions
- blockers
- relevant files
- attempts / failures
- artifact refs

Outcome:
- DeepClaw gains real task memory that is not just transcript residue

---

## Phase 3 — Add continuity schema and rebuild-from-state path
Build a continuity representation derived from working state.

Use it for:
- resuming long tasks
- rebuilding prompts when transcript quality degrades
- preserving anti-repetition notes
- fallback after compaction

Outcome:
- continuity becomes state-driven, not transcript-driven

---

## Phase 4 — Add explicit route selection
Implement a route selector before model invocation:
- fits
- shrink_artifacts
- compact_history
- rebuild_from_state
- boundary_reset

Outcome:
- DeepClaw stops treating “whatever is in the thread right now” as the only available prompt source

---

## Phase 5 — Make local context selective
Refactor `LocalContextMiddleware` so it can emit:
- stable baseline context
- optional task-specific enrichments
- smaller file/repo views for non-coding tasks

Outcome:
- local context becomes higher-signal and cheaper

---

## Phase 6 — Add explicit boundary operations
Only after the above, add richer boundary tools if still needed:
- manual `/compact`
- model-switch boundary mode
- richer thread metadata
- continuity export/import

Outcome:
- thread boundaries become deliberate operator tools, not compensatory architecture

---

## Required behavioral guarantees

1. **Transcript is not the primary working memory**
2. **Large artifacts are not repeatedly stuffed back into prompt context**
3. **Working state survives long sessions better than raw transcript alone**
4. **Compaction does not destroy recoverability**
5. **Rebuild-from-state can continue work without major goal drift**
6. **Summaries/continuity state are non-authoritative and reference-oriented**
7. **Memory, skills, state, transcript, and artifacts remain distinct layers**

---

## Final recommendation

The fresh post-PR-30 design is:

> **Keep the DeepAgents summarization/offload/local-context substrate. Add a DeepClaw-owned structured working-state layer and route selector above it. Treat thread rotation as a fallback boundary mechanism, not as the main context architecture.**

That is the cleanest path from “better transcript hygiene” to a genuinely strong long-horizon technical agent.
