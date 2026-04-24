"""Derived local/project context helpers for coding sessions."""

from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.middleware.memory import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ResponseT

from deepclaw.runtime_hygiene import get_runtime_state

_PROJECT_MARKERS = {
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Makefile",
}
_MONOREPO_MARKERS = {
    "pnpm-workspace.yaml": "pnpm workspace",
    "turbo.json": "turbo monorepo",
    "nx.json": "nx workspace",
    "lerna.json": "lerna monorepo",
}
_CONTEXT_SECTION_TITLE = "## Local Project Context"
_RUNTIME_METADATA_KEY = "derived_project_context_block"


@dataclass(slots=True)
class DerivedProjectContext:
    """Compact derived facts about the local coding workspace."""

    cwd: Path
    project_root: Path | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    languages: tuple[str, ...] = ()
    package_managers: tuple[str, ...] = ()
    shape: str | None = None

    def render_block(self) -> str:
        """Render a compact prompt block."""
        lines = [
            _CONTEXT_SECTION_TITLE,
            f"- cwd: `{self.cwd}`",
        ]
        if self.project_root is not None and self.project_root != self.cwd:
            lines.append(f"- project root: `{self.project_root}`")
        if self.git_branch:
            status = "dirty" if self.git_dirty else "clean"
            lines.append(f"- git branch: `{self.git_branch}` ({status})")
        if self.languages:
            lines.append(f"- language/runtime: {', '.join(self.languages)}")
        if self.package_managers:
            lines.append(f"- package manager: {', '.join(self.package_managers)}")
        if self.shape:
            lines.append(f"- project shape: {self.shape}")
        return "\n".join(lines)


def _normalize_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _run_git(path: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output or None


def _detect_git_root(path: Path) -> Path | None:
    top_level = _run_git(path, "rev-parse", "--show-toplevel")
    return Path(top_level).resolve() if top_level else None


def _detect_git_branch(path: Path) -> str | None:
    return _run_git(path, "rev-parse", "--abbrev-ref", "HEAD")


def _detect_git_dirty(path: Path) -> bool:
    status = _run_git(path, "status", "--short")
    return bool(status)


def _is_project_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / ".git").exists():
        return True
    return any((path / marker).exists() for marker in _PROJECT_MARKERS | set(_MONOREPO_MARKERS))


def _child_project_dirs(path: Path) -> list[Path]:
    children: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        if _is_project_dir(child):
            children.append(child.resolve())
    return children


def _load_pyproject(path: Path) -> dict[str, Any]:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    try:
        with pyproject.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _detect_languages(path: Path) -> tuple[str, ...]:
    languages: list[str] = []
    if (path / "pyproject.toml").is_file() or (path / "requirements.txt").is_file():
        languages.append("Python")
    if (path / "package.json").is_file():
        languages.append("TypeScript" if (path / "tsconfig.json").is_file() else "Node.js")
    if (path / "Cargo.toml").is_file():
        languages.append("Rust")
    if (path / "go.mod").is_file():
        languages.append("Go")
    if (path / "Gemfile").is_file():
        languages.append("Ruby")
    if (
        (path / "pom.xml").is_file()
        or (path / "build.gradle").is_file()
        or (path / "build.gradle.kts").is_file()
    ):
        languages.append("Java")
    return tuple(languages)


def _detect_package_managers(path: Path) -> tuple[str, ...]:
    managers: list[str] = []
    pyproject = _load_pyproject(path)
    pyproject_tool = pyproject.get("tool", {}) if isinstance(pyproject, dict) else {}

    if (path / "uv.lock").is_file() or "uv" in pyproject_tool:
        managers.append("uv")
    elif (path / "poetry.lock").is_file() or "poetry" in pyproject_tool:
        managers.append("poetry")
    elif (path / "pdm.lock").is_file() or "pdm" in pyproject_tool:
        managers.append("pdm")
    elif (path / "requirements.txt").is_file() or (path / "pyproject.toml").is_file():
        managers.append("pip")

    if (path / "pnpm-lock.yaml").is_file() or (path / "pnpm-workspace.yaml").is_file():
        managers.append("pnpm")
    elif (path / "yarn.lock").is_file():
        managers.append("yarn")
    elif (path / "package-lock.json").is_file() or (path / "package.json").is_file():
        managers.append("npm")

    if (path / "Cargo.lock").is_file() or (path / "Cargo.toml").is_file():
        managers.append("cargo")
    if (path / "go.mod").is_file():
        managers.append("go")
    if (path / "Gemfile.lock").is_file() or (path / "Gemfile").is_file():
        managers.append("bundler")
    if (path / "pom.xml").is_file():
        managers.append("maven")
    if (path / "build.gradle").is_file() or (path / "build.gradle.kts").is_file():
        managers.append("gradle")

    return tuple(managers)


def _detect_shape(cwd: Path, project_root: Path | None) -> str | None:
    root = project_root or cwd
    for marker, label in _MONOREPO_MARKERS.items():
        if (root / marker).is_file():
            return label

    project_children = _child_project_dirs(root)
    if project_root is None:
        if not project_children:
            return None
        if len(project_children) == 1:
            return "single-project workspace"
        sample = ", ".join(child.name for child in project_children[:4])
        suffix = "" if len(project_children) <= 4 else ", ..."
        return f"multi-project workspace ({sample}{suffix})"

    nested_children = [child for child in project_children if child != project_root]
    if len(nested_children) >= 2:
        sample = ", ".join(child.name for child in nested_children[:4])
        suffix = "" if len(nested_children) <= 4 else ", ..."
        return f"monorepo ({sample}{suffix})"

    if _detect_git_root(root) is not None:
        return "single-project repo"
    return "single-project workspace"


def derive_project_context(workspace_root: str | Path) -> DerivedProjectContext | None:
    """Derive compact local/project facts from the configured workspace root."""
    cwd = _normalize_root(workspace_root)
    if not cwd.exists() or not cwd.is_dir():
        return None

    project_root = _detect_git_root(cwd)
    if project_root is None and _is_project_dir(cwd):
        project_root = cwd
    if project_root is None:
        candidates = _child_project_dirs(cwd)
        if len(candidates) == 1:
            project_root = candidates[0]

    info_root = project_root or cwd
    git_root = _detect_git_root(info_root)
    git_branch = _detect_git_branch(info_root) if git_root is not None else None
    git_dirty = _detect_git_dirty(info_root) if git_root is not None else False

    return DerivedProjectContext(
        cwd=cwd,
        project_root=project_root,
        git_branch=git_branch,
        git_dirty=git_dirty,
        languages=_detect_languages(info_root),
        package_managers=_detect_package_managers(info_root),
        shape=_detect_shape(cwd, project_root),
    )


def build_derived_project_context_block(workspace_root: str | Path) -> str | None:
    """Build the compact local/project context block for prompt injection."""
    context = derive_project_context(workspace_root)
    return context.render_block() if context is not None else None


class DerivedProjectContextMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Inject compact derived local/project facts into the system prompt."""

    def __init__(self, *, workspace_root: str | Path) -> None:
        self.workspace_root = workspace_root

    def _get_context_block(self) -> str | None:
        runtime_state = get_runtime_state()
        if runtime_state is not None:
            cached = runtime_state.metadata.get(_RUNTIME_METADATA_KEY)
            if isinstance(cached, str):
                return cached

        block = build_derived_project_context_block(self.workspace_root)
        if runtime_state is not None and block is not None:
            runtime_state.metadata[_RUNTIME_METADATA_KEY] = block
        return block

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Append derived project context to the system prompt when available."""
        context_block = self._get_context_block()
        if not context_block:
            return request

        new_system_message = append_to_system_message(request.system_message, context_block)
        return request.override(system_message=new_system_message)

    def wrap_model_call(self, request, handler):  # type: ignore[override]
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(self, request, handler):  # type: ignore[override]
        modified_request = self.modify_request(request)
        return await handler(modified_request)
