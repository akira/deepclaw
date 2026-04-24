# DeepClaw Context Compaction Implementation Plan

**Goal:** Add a first-class DeepClaw compaction/thread-rotation layer with richer thread metadata, explicit `/compact`, offload-first historical artifacts, and summary injection into fresh threads.

**Architecture:** Keep DeepAgents' built-in summarization middleware intact, but add a DeepClaw-owned orchestration layer around thread rotation and compaction. Store richer per-chat thread metadata, generate recoverable summary/raw-history artifacts, and trigger compaction manually (`/compact`) and automatically before oversized thread invocations.

**Tech Stack:** Python 3.14, DeepClaw Telegram gateway, LangGraph AsyncSqliteSaver checkpoints, pytest.

---

## Task 1: Add compaction metadata and persistence helpers
- Create `deepclaw/compaction.py`
- Modify `deepclaw/auth.py`
- Add tests in `tests/test_bot.py`

Implement:
- `ThreadState` / metadata helpers with backward compatibility for legacy `chat_id -> thread_id` JSON
- summary artifact/raw-history artifact path helpers
- load/save helpers for richer thread metadata

Verification:
- tests for legacy thread-id loading
- tests for round-tripping new metadata structure

## Task 2: Implement checkpoint inspection + artifact generation
- Create logic in `deepclaw/compaction.py`
- Read checkpoint messages via `aget_tuple`
- Produce:
  - raw history markdown artifact
  - structured handoff summary text
  - fresh thread id

Verification:
- unit tests with mocked checkpoint tuple
- assert raw-history and summary files are created and contain expected content

## Task 3: Wire manual `/compact` command into Telegram
- Modify `deepclaw/channels/telegram.py`
- Register handler and bot command
- Update `/help`

Behavior:
- compact current chat thread into artifacts
- rotate to fresh thread
- store pending summary injection for next inbound turn
- reply with confirmation including new thread id

Verification:
- slash command tests for `/compact`
- thread metadata updated
- confirmation text includes compaction result

## Task 4: Add automatic pre-invoke compaction in gateway
- Modify `deepclaw/gateway.py`
- Pass checkpointer + thread state store into `Gateway`
- Before `astream(...)`, inspect thread size and compact when threshold exceeded
- Inject reference-only summary text into the first message of the new thread

Verification:
- gateway test with mocked compaction result
- ensure injected user content includes reference-only summary preamble

## Task 5: Rebuild gateway/model switch integration
- Modify `deepclaw/channels/telegram.py`
- Ensure `Gateway(...)` receives checkpointer + thread-state store in `post_init()` and `/model`
- On `/model` switch, prompt user to use `/compact` or auto-rotate if needed

Verification:
- `/model` tests updated for new gateway constructor
- no regressions in existing slash-command tests

## Task 6: Include design doc in branch
- Add `docs/deepclaw-context-engineering-redesign.md`
- Ensure PR body references it explicitly

## Task 7: Run tests and open PR
- Run targeted pytest first
- Run full pytest suite
- Commit with doc + code
- Push branch to `fork`
- Open PR against `akira/deepclaw`
