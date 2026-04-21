---
name: devin-review
description: Run Devin AI code review on a GitHub PR, interpret findings, and fix flagged bugs iteratively.
triggers:
  - "run devin review"
  - "check devin review"
  - "devin code review on PR"
---

# Devin Review Workflow

## What it is
Devin Review (https://app.devin.ai/review) is a PR viewer + AI analyzer. Not a GitHub App — you paste a PR URL to trigger analysis. Works on public and private repos without login.

## Steps

### 1. Open and submit PR URL
- Navigate to https://app.devin.ai/review
- Type the PR URL into the textbox (e.g. https://github.com/owner/repo/pull/N)
- Click Submit — the button enables after typing
- A modal appears: "Devin is analyzing your PR... We'll let you know when it's ready"

### 2. Wait for analysis
- Analysis takes 60-90 seconds
- Poll with browser_vision every 30s asking "Is analysis complete? What does the sidebar show?"
- When done, a "Refresh to view latest commit" dialog appears — click Refresh

### 3. Read findings
- Right sidebar shows Bugs (red) and Flags (orange)
- Bugs = real issues to fix; Flags = areas to review carefully
- Vision prompt: "What bugs and flags does the right sidebar show? List all with descriptions and line numbers."

### 4. Re-analyze after fixes
- After pushing fixes, go back to https://app.devin.ai/review, paste PR URL again
- A dialog appears: "Generate new analysis for latest commit [sha] Not analyzed"
- Click Generate — new analysis runs on the updated commit
- When done, click Refresh to load results
- Previously fixed bugs show as "Resolved" (gray) in the sidebar

## Pitfalls
- The page caches old analysis — always re-submit the URL after new commits, don't just reload
- Vision analysis sometimes reports stale line numbers if the modal is covering the sidebar
- "Checks: Partial 3/4" in sidebar is normal, not an error
- Flags at old line numbers after fixes = stale view, need to Generate new analysis
- Analysis takes longer on subsequent commits (2-3 min) — poll every 60s not 30s
- Sidebar bug count can *increase* between iterations as Devin notices new issues uncovered by fixes (e.g. fixing /memory path revealed HeartbeatRunner also needed updating)
- Resolved bugs from prior commits remain visible (grayed out) — only count open (red) bugs as actionable
- The CONFIG_KEY overwrite flag is a persistent Devin concern about runtime state not surviving restarts — it's a design note, not a blocking bug; safe to leave unfixed if intentional

## Iterative fix loop
1. Fix all bugs Devin flagged
2. Run the project test suite and confirm passing
3. Commit and push to the fork branch
4. Re-run Devin Review on the PR URL
5. Repeat until sidebar shows 0 bugs (flags are advisory)
