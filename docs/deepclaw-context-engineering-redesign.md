# DeepClaw Context Engineering Redesign

## Goal

Redesign DeepClaw's context strategy so it behaves like a deliberately layered agent system instead of a long-lived sticky thread.

Target layers:
1. always-on prompt
2. durable memory
3. durable skills
4. active raw working context
5. compressed historical context
6. offloaded artifacts
7. context isolation via subagents

The goal is to preserve continuity for long-running development work **without** letting checkpointed thread state become the system's de facto long-term memory.

---

## Executive Summary

DeepClaw already has strong foundations:
- `SOUL.md` for identity and behavior
- `AGENTS.md` memory via `MemoryMiddleware`
- skills via `SkillsMiddleware`
- persistent checkpoint state via LangGraph / DeepAgents
- built-in subagents for isolation
- inherited DeepAgents summarization middleware through `create_deep_agent(...)`

The redesign is not about replacing these layers. It is about making the overall context lifecycle explicit and better aligned with the newer source set:

- **LangChain / Deep Agents**: compression should be a **ladder**, not one blunt summarization step — offload large tool results first, then redundant tool inputs, then summarize only when needed.
- **Anthropic**: context is a finite attention budget; long-horizon systems need compaction, structured note-taking, and sub-agent architectures. Compaction quality depends on preserving the right details and discarding the right noise.
- **Galileo**: context and memory must be treated as different architectural layers. Context is working memory; memory is external long-term storage.
- **OpenAI**: long-term personalization works best when memory is treated as **structured state** with scoped ownership, in-run distillation, post-run consolidation, and selective reinjection.

So the redesign move is:

> Keep DeepClaw's prompt, memory, skills, and subagent foundations, but add an explicit DeepClaw-owned compaction and thread-rotation layer that treats summaries as reference-only handoffs, uses offloading aggressively, and shrinks the active thread into a bounded working set.

---

## Design Principles

### 1. Do not treat one checkpointed thread as the primary memory system
A long-lived LangGraph thread is a useful transport for short-term state, but it should not be the authoritative store for everything the system has ever learned or done.

### 2. Keep active context small, high-signal, and current
Recent turns should stay raw. Older resolved turns should be compressed, offloaded, or moved into more appropriate durable layers.

### 3. Use a compression ladder, not a single compression hammer
Follow the order suggested by the LangChain / Deep Agents context-management article:
1. offload large tool results first
2. truncate/offload redundant historical tool inputs next
3. summarize only after cheaper compression opportunities are exhausted

### 4. Make recoverability a first-class requirement
Compression is only safe if the system can recover details that were summarized away. Summaries and offloads must preserve references to canonical raw artifacts.

### 5. Separate working context from long-term memory
- **Context** = working memory, expensive, limited, volatile
- **Memory** = persistent storage, cheap, large, retrieved selectively

### 6. Prefer structured state over prompt accretion
For durable facts and continuity, prefer note/state objects, memory files, and skills over endlessly appending more history or rules into the prompt.

### 7. Treat summaries as historical reference, not live instructions
A compressed handoff must be clearly marked as background/reference. It should not feel like the top of the active instruction stack.

---

## Context Layers

## Layer A — Always-on prompt
**Purpose:** stable identity and invariant operating heuristics.

**Contents:**
- `SOUL.md`
- tool-use enforcement
- minimal stable DeepClaw operating rules

**Should contain:**
- identity
- tone
- autonomy norms
- high-level tool discipline

**Should not contain:**
- large project notes
- historical task detail
- durable facts that belong in memory
- reusable workflows that belong in skills
- temporary task state

Anthropic's guidance about prompt altitude applies here: the prompt should be specific enough to guide behavior, but not so overloaded that it becomes a brittle rules engine.

---

## Layer B — Durable memory
**Purpose:** facts that should persist across conversations.

**Contents:**
- `~/.deepclaw/AGENTS.md`
- stable preferences
- conventions
- recurring environment facts
- durable user/project facts

**Access pattern:**
- loaded through `MemoryMiddleware`
- must invalidate cached thread state after writes

Memory should behave more like a structured persistent state layer than “old conversation residue.” The OpenAI memory lifecycle is useful here: capture candidate facts during the run, consolidate them deliberately, and inject only the relevant slice next time.

---

## Layer C — Durable skills
**Purpose:** procedural memory and reusable workflows.

**Contents:**
- `~/.deepclaw/skills/*/SKILL.md`
- bundled skills
- reusable multi-step instructions

**Access pattern:**
- skill metadata visible via `SkillsMiddleware`
- full skill content loaded on demand when relevant

This is a form of progressive disclosure. Keep startup context lean by surfacing only enough information to decide whether a skill is relevant, then load the detailed procedure only when needed.

---

## Layer D — Active raw working context
**Purpose:** the current task's live working set.

**Contents:**
- recent user/assistant turns
- unresolved current task details
- recent tool activity still relevant to the next step
- current file/artifact references
- active warnings or error state

**Policy:**
- keep only a bounded recent window raw
- do not keep resolved old turns indefinitely
- do not allow stale tool chatter to dominate the active context
- do not confuse “thread has it somewhere” with “model should still attend to it now”

This layer is the real equivalent of working memory / RAM. It should stay focused and small enough to remain sharp.

---

## Layer E — Compressed historical context
**Purpose:** preserve continuity without keeping all raw history active.

**Contents:**
A structured handoff summary artifact containing, at minimum:
- session intent / active objective
- key decisions
- unresolved bugs or open questions
- pending work / next steps
- relevant files and artifacts
- important recent failures
- things already tried
- risks / anti-repetition notes

**Critical property:**
This summary must be clearly labeled as:
- background/reference only
- not active instructions
- compressed historical context

Source-set guidance:
- Anthropic: compaction should preserve architectural decisions, unresolved bugs, implementation detail, and enough state to continue long-horizon work.
- LangChain / Deep Agents: the in-context summary is paired with filesystem preservation of raw history.
- Galileo: summaries must not become a junk-drawer substitute for real architecture.

---

## Layer F — Offloaded artifacts
**Purpose:** move bulky raw material out of active context while preserving recoverability.

**Contents:**
- large tool outputs
- raw logs
- raw transcripts
- old thread archives
- code-search dumps
- fetched documents
- old file-edit payloads
- canonical raw conversation archives

**Policy:**
- reference by path/id instead of reinjecting inline
- keep previews small
- allow search / `read_file` / retrieval when needed
- summaries should point to artifacts rather than repeat raw material

Following the LangChain article, offloading should happen **before** summarization wherever possible.

---

## Layer G — Context isolation via subagents
**Purpose:** keep exploratory or heavy work from polluting the main thread.

Use subagents for:
- research
- coding subtasks
- sysadmin diagnosis

The main agent should receive compressed results, not all raw exploration.

---

## Compaction Model

## Trigger conditions
Compaction should trigger when one or more of the following is true:
- estimated token count exceeds threshold
- message count exceeds threshold
- large tool outputs or edit arguments accumulate
- thread age exceeds threshold
- model switch occurs via `/model`
- a major topic boundary is detected
- recent tool chatter dominates the active context

## Compression ladder
Compression should follow this order:

### Step 1 — Offload large tool results
If a tool result is individually huge, move it to external storage immediately and keep only:
- path/reference
- short preview
- enough metadata to find it again

### Step 2 — Truncate/offload stale tool inputs
For historical write/edit calls whose payloads already exist on disk, replace old raw arguments with references to canonical file paths.

### Step 3 — Summarize older conversation history
Only when offloading no longer creates enough room should the system generate a structured handoff summary.

This is safer than jumping straight to broad summarization.

## Compaction flow
When compaction is triggered:
1. select recent raw window
2. identify offload candidates
3. offload first
4. summarize older remaining history if needed
5. generate a structured handoff artifact
6. rotate to a fresh thread
7. persist lineage metadata

---

## Recoverability requirements
Compression is only acceptable if the system can recover details later.

Required tests:
1. Can the agent continue the task after compaction without goal drift?
2. Can it recover a summarized-away detail from offloaded artifacts?
3. Does it avoid re-asking already answered questions after compaction?
4. Does it avoid falsely declaring the task complete after compaction?
5. Does it preserve unresolved bugs, active objective, and next steps?

---

## Thread operation modes
### `/new`
Start a fresh thread with no carryover except durable memory and skills.

### `/compact`
Start a fresh thread with:
- structured handoff summary
- links to offloaded artifacts
- explicit parent/child lineage

### `/clear`
Hard reset transient thread state.

---

## Required Behavioral Guarantees

1. **Summaries are non-authoritative**
2. **Memory and skills are not silently stale**
3. **Model switches do not inherit unsafe stale context blindly**
4. **Large artifacts are not repeatedly stuffed back into context**
5. **Active context remains bounded**
6. **Recoverability is preserved**
7. **Context and memory remain distinct layers**

---

## Code Paths to Patch First

## 1. `deepclaw/channels/telegram.py`
Relevant functions:
- `get_thread_id(...)`
- `cmd_new(...)`
- `cmd_clear(...)`
- `cmd_model(...)`

Changes:
- add a thread-rotation helper supporting fresh / compacted / cleared modes
- stop treating `/clear` as the only recommended response to model switches
- enrich thread metadata beyond `chat_id -> thread_id`
- add model-switch-aware compaction path

## 2. `deepclaw/gateway.py`
Relevant function:
- `Gateway.handle_message(...)`

Changes:
- add pre-invoke compaction checks
- estimate thread size / token pressure before calling `astream(...)`
- offload large/stale raw artifacts before broad summarization
- inject compaction metadata cleanly when applicable
- ensure old tool chatter does not become the de facto long-term context backbone

## 3. `deepclaw/agent.py`
Relevant function:
- `create_agent(...)`

Changes:
- explicitly document DeepClaw's summarization and compaction assumptions
- preserve existing DeepAgents summarization middleware unless strong evidence suggests replacement
- add DeepClaw-specific compaction-related middleware only with clear scope
- strengthen memory/skills cache invalidation policy

## 4. `deepclaw/auth.py`
Changes:
- evolve storage from plain thread-id mapping to richer thread/session metadata
- support lineage, compaction references, summary artifact tracking, and parent/child thread relationships

## 5. New module: `deepclaw/compaction.py`
Responsibilities:
- inspect current thread state
- estimate compaction need
- identify offload candidates
- offload bulky artifacts
- generate structured handoff summary
- rotate thread
- persist lineage metadata
- return compaction result object

---

## Suggested Implementation Phases

## Phase 1 — Instrumentation
Log:
- message counts
- estimated token counts
- size of checkpointed state
- when summarization middleware triggers
- when memory/skills metadata remains cached
- what proportion of active context is tool chatter or raw artifacts

## Phase 2 — Compaction orchestrator
Create `deepclaw/compaction.py` with:
- threshold checks
- offload-first logic
- summary builder
- thread rotation
- metadata persistence
- recoverability references

## Phase 3 — Telegram / gateway integration
Patch:
- `telegram.py`
- `gateway.py`

to support:
- automatic compaction on large threads
- compact-on-model-switch
- explicit fresh / compact / clear operations
- developer-visible compaction logging

## Phase 4 — Cache invalidation hardening
Ensure:
- memory writes invalidate `memory_contents`
- skill writes invalidate `skills_metadata`
- invalidation happens after successful mutating operations and slash commands

## Phase 5 — Recoverability tests and targeted evals
Add focused tests that deliberately trigger compaction and check:
- objective preservation
- recoverability of summarized-away details
- absence of goal drift
- absence of false completion
- correct artifact references

---

## Recommendations

1. Keep the current prompt/memory/skills/subagent architecture.
2. Do not blindly stack a second summarizer on top of DeepAgents. Build a DeepClaw-owned **compaction orchestrator** around the existing framework summarization machinery.
3. Implement the **compression ladder** explicitly.
4. Adopt **recoverability** as the design test for compaction.
5. Treat memory more like a **structured persistent state layer** and less like “old stuff in the thread.”
6. Use subagents more aggressively for context isolation in exploratory work.
7. Reduce the authority of long-lived checkpoint threads; increase the authority of explicit layered context.

---

## Final Assessment

DeepClaw's current context engineering strategy is promising but incomplete.

It is already strong in:
- identity
- durable memory
- skills
- subagents

It is still weak in:
- explicit compaction semantics
- recovery-oriented offloading
- summary / handoff design
- bounded active context discipline
- clean separation between short-term thread state and long-term memory

The redesign therefore is not “add summarization middleware.” It is:

> Turn compression, offloading, memory separation, and thread rotation into an explicit DeepClaw-owned context architecture.

That is the path from a sticky-thread system to a reliable long-horizon development agent.
