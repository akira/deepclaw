---
name: github-pr-workflow
description: DeepClaw-specific pull request workflow — branch from akira/deepclaw main, run the project venv tests, push to the BlueMeadow19 fork, open/update PRs, and manage CI follow-up.
version: 1.2.0
author: DeepClaw
license: MIT
metadata:
  hermes:
    tags: [GitHub, Pull-Requests, CI/CD, DeepClaw, Git]
    related_skills: [github-code-review, deepclaw-development]
---

# DeepClaw GitHub PR Workflow

Use this skill for the **actual DeepClaw repo workflow** on this machine.

## Scope

This skill is specific to:
- repo: `/home/ubuntu/deepclaw`
- upstream remote: `origin` → `git@github.com:akira/deepclaw.git`
- fork remote: `fork` → `git@github.com:BlueMeadow19/deepclaw.git`
- test environment: `/home/ubuntu/deepclaw/.venv`

Do **not** use generic placeholder paths like `src/auth.py` or sample app PR text. Ground everything in the DeepClaw repo.

## Deterministic First

Before changing PR state, always confirm:
1. current branch
2. current diff vs `origin/main`
3. remotes and target repo
4. whether there is already an open PR for the branch
5. whether local `main` contains deploy-only merges that should not leak into the PR

Recommended checks:

```bash
cd /home/ubuntu/deepclaw
git status --short --branch
git remote -v | cat
git fetch origin fork
git log --oneline --decorate -n 8
gh pr list --repo akira/deepclaw --state open --limit 20
```

## Branching Rule

DeepClaw local `main` may contain deploy-only merges. When creating a PR branch, branch from **`origin/main`**, not blindly from local `main`.

```bash
cd /home/ubuntu/deepclaw
git fetch origin
git checkout -b feat/my-change origin/main
```

If the branch already exists and is polluted, clean it by resetting to `origin/main` and cherry-picking only the intended commits.

## Commit Workflow

Stage only the intended files. Never include `.beads/`.

```bash
cd /home/ubuntu/deepclaw
git add deepclaw/... tests/... pyproject.toml uv.lock
git commit -m "feat: concise deepclaw change summary"
```

Conventional commit types commonly used here:
- `feat:` new capability
- `fix:` bug fix
- `refactor:` non-behavioral restructuring
- `docs:` documentation-only change
- `test:` test-only change
- `chore:` maintenance / dependency bump

## Required Verification Before Push

Use the **DeepClaw venv Python directly**.

### Full suite
```bash
/home/ubuntu/deepclaw/.venv/bin/python -m pytest tests/ -v --tb=short
```

### Focused suite
```bash
/home/ubuntu/deepclaw/.venv/bin/python -m pytest tests/test_bot.py -v
```

### Ruff
```bash
cd /home/ubuntu/deepclaw
ruff check .
ruff format --check .
```

Important: `ruff check` and `ruff format --check` are separate gates. Run both.

## Push + Open PR

DeepClaw PRs go to **`akira/deepclaw`** from the **`BlueMeadow19/deepclaw`** fork.

```bash
cd /home/ubuntu/deepclaw
git push -u fork HEAD
gh pr create \
  --repo akira/deepclaw \
  --head BlueMeadow19:$(git branch --show-current) \
  --base main \
  --title "feat: concise deepclaw change summary" \
  --body-file /tmp/deepclaw-pr-body.md
```

Safe PR-body pattern:

```bash
cat >/tmp/deepclaw-pr-body.md <<'EOF'
## Summary
- explain the DeepClaw-specific change
- mention key files touched

## Testing
- `/home/ubuntu/deepclaw/.venv/bin/python -m pytest tests/...`
- `ruff check .`
- `ruff format --check .`
EOF
```

## Updating Existing PRs

Check whether the current branch already has an open PR:

```bash
gh pr status --repo akira/deepclaw
```

If yes, push new commits to the same branch:

```bash
git push fork HEAD
```

## CI Follow-Up

```bash
gh pr checks --repo akira/deepclaw --watch
```

Or for a specific PR:

```bash
gh pr checks <PR_NUMBER> --repo akira/deepclaw --watch
```

If CI fails:
1. inspect the failing job
2. patch the code/tests
3. re-run the relevant local checks
4. commit + push again

Useful commands:

```bash
gh run list --repo akira/deepclaw --branch $(git branch --show-current) --limit 5
gh run view <RUN_ID> --repo akira/deepclaw --log-failed
```

## DeepClaw-Specific Pitfalls

### 1. `gh pr view` / `gh pr edit` GraphQL classic-projects failure
On this repo, some `gh pr` commands can fail with a Projects (classic) GraphQL error. If that happens, use `gh api` instead.

Example:

```bash
gh api repos/akira/deepclaw/pulls/<PR_NUMBER>
```

For updates, prepare JSON and PATCH through `gh api` rather than relying on `gh pr edit`.

### 2. Use the project venv, not Hermes's environment
Bad:
```bash
uv run pytest
```

Good:
```bash
/home/ubuntu/deepclaw/.venv/bin/python -m pytest tests/ -v --tb=short
```

### 3. Keep `.beads/` out of PRs
If accidentally staged:

```bash
git rm --cached .beads/issues.jsonl
git add .gitignore
git commit --amend --no-edit
git push --force-with-lease fork HEAD
```

### 4. Distinguish PR work from local deploy work
A GitHub PR should reflect the code change only. Local deploy merges like:
- `deploy: merge ... locally`
should stay out of the PR history.

## After Merge / Local Deploy

If the user wants the merged code deployed locally after PR work is ready:

```bash
cd /home/ubuntu/deepclaw
git checkout main
git pull --ff-only origin main
systemctl --user restart deepclaw.service
systemctl --user status deepclaw.service --no-pager
journalctl --user -u deepclaw.service -n 30 --no-pager
```

Do this only when the task explicitly includes deploy/restart, not for every PR by default.

## Review Checklist for This Workflow

Before saying PR work is complete, verify:
- branch is based correctly
- intended files only are included
- tests/ruff were run with the DeepClaw venv
- branch pushed to `fork`
- PR exists on `akira/deepclaw`
- CI status was checked
- any needed follow-up comments/updates were made

## Minimal DeepClaw PR Template

```markdown
## Summary
- what changed in DeepClaw
- why it changed

## Testing
- exact commands run from `/home/ubuntu/deepclaw`

## Notes
- any deploy implications
- any known follow-up work
```
