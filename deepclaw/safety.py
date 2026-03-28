"""Dangerous command detection for use with DeepAgents' interrupt_on mechanism.

Provides regex-based pattern matching to identify dangerous shell commands,
destructive SQL statements, and other risky operations before they execute.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DangerousPattern:
    """A pattern that matches a dangerous command."""

    pattern: re.Pattern[str]
    category: str
    description: str
    severity: str  # "critical" or "warning"


# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

CATEGORY_RECURSIVE_DELETE = "recursive_delete"
CATEGORY_FILESYSTEM_FORMAT = "filesystem_format"
CATEGORY_PERMISSION_CHANGE = "permission_change"
CATEGORY_SQL_DESTRUCTION = "sql_destruction"
CATEGORY_SYSTEM_CONFIG_WRITE = "system_config_write"
CATEGORY_SERVICE_MANAGEMENT = "service_management"
CATEGORY_MASS_PROCESS_KILL = "mass_process_kill"
CATEGORY_FORK_BOMB = "fork_bomb"
CATEGORY_PIPED_EXECUTION = "piped_execution"
CATEGORY_CODE_INJECTION = "code_injection"
CATEGORY_DESTRUCTIVE_FIND = "destructive_find"
CATEGORY_DESTRUCTIVE_GIT = "destructive_git"
CATEGORY_DEVICE_WRITE = "device_write"

# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS: list[DangerousPattern] = [
    # Recursive deletion
    DangerousPattern(
        pattern=re.compile(r"\brm\s+.*-\w*r\w*f", re.IGNORECASE),
        category=CATEGORY_RECURSIVE_DELETE,
        description="Recursive forced deletion (rm -rf)",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\brm\s+.*-\w*r\b", re.IGNORECASE),
        category=CATEGORY_RECURSIVE_DELETE,
        description="Recursive deletion (rm -r)",
        severity="critical",
    ),
    # Filesystem formatting
    DangerousPattern(
        pattern=re.compile(r"\bmkfs\b"),
        category=CATEGORY_FILESYSTEM_FORMAT,
        description="Filesystem formatting (mkfs)",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bdd\s+if="),
        category=CATEGORY_FILESYSTEM_FORMAT,
        description="Low-level disk copy (dd if=)",
        severity="critical",
    ),
    # Permission changes
    DangerousPattern(
        pattern=re.compile(r"\bchmod\s+777\b"),
        category=CATEGORY_PERMISSION_CHANGE,
        description="World-writable permissions (chmod 777)",
        severity="warning",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bchmod\s+.*-R\b"),
        category=CATEGORY_PERMISSION_CHANGE,
        description="Recursive permission change (chmod -R)",
        severity="warning",
    ),
    # SQL destruction
    DangerousPattern(
        pattern=re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
        category=CATEGORY_SQL_DESTRUCTION,
        description="SQL table drop (DROP TABLE)",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bDROP\s+DATABASE\b", re.IGNORECASE),
        category=CATEGORY_SQL_DESTRUCTION,
        description="SQL database drop (DROP DATABASE)",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bDELETE\s+FROM\s+\w+\s*;", re.IGNORECASE),
        category=CATEGORY_SQL_DESTRUCTION,
        description="DELETE FROM without WHERE clause",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
        category=CATEGORY_SQL_DESTRUCTION,
        description="SQL table truncation (TRUNCATE)",
        severity="critical",
    ),
    # System config writes
    DangerousPattern(
        pattern=re.compile(r">\s*/etc/"),
        category=CATEGORY_SYSTEM_CONFIG_WRITE,
        description="Writing to /etc/ (system configuration)",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r">\s*/boot/"),
        category=CATEGORY_SYSTEM_CONFIG_WRITE,
        description="Writing to /boot/ (boot configuration)",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r">\s*/usr/"),
        category=CATEGORY_SYSTEM_CONFIG_WRITE,
        description="Writing to /usr/ (system binaries)",
        severity="critical",
    ),
    # Service management
    DangerousPattern(
        pattern=re.compile(r"\bsystemctl\s+(stop|disable)\b"),
        category=CATEGORY_SERVICE_MANAGEMENT,
        description="Stopping or disabling a systemd service",
        severity="warning",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bservice\s+\S+\s+stop\b"),
        category=CATEGORY_SERVICE_MANAGEMENT,
        description="Stopping a service via init.d",
        severity="warning",
    ),
    # Mass process killing
    DangerousPattern(
        pattern=re.compile(r"\bkill\s+-9\s+-1\b"),
        category=CATEGORY_MASS_PROCESS_KILL,
        description="Kill all user processes (kill -9 -1)",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bpkill\s+-9\b"),
        category=CATEGORY_MASS_PROCESS_KILL,
        description="Force kill processes by pattern (pkill -9)",
        severity="warning",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bkillall\b"),
        category=CATEGORY_MASS_PROCESS_KILL,
        description="Kill all processes by name (killall)",
        severity="warning",
    ),
    # Fork bombs
    DangerousPattern(
        pattern=re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;\s*:"),
        category=CATEGORY_FORK_BOMB,
        description="Fork bomb detected",
        severity="critical",
    ),
    # Piped execution
    DangerousPattern(
        pattern=re.compile(r"\b(curl|wget)\b.*\|\s*(sh|bash)\b"),
        category=CATEGORY_PIPED_EXECUTION,
        description="Piped remote script execution (curl/wget | sh/bash)",
        severity="critical",
    ),
    # Code injection vectors
    DangerousPattern(
        pattern=re.compile(r"\bpython\s+-c\b"),
        category=CATEGORY_CODE_INJECTION,
        description="Inline Python execution (python -c)",
        severity="warning",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bbash\s+-c\b"),
        category=CATEGORY_CODE_INJECTION,
        description="Inline bash execution (bash -c)",
        severity="warning",
    ),
    DangerousPattern(
        pattern=re.compile(r"\beval\b"),
        category=CATEGORY_CODE_INJECTION,
        description="Dynamic code evaluation (eval)",
        severity="warning",
    ),
    # Destructive find
    DangerousPattern(
        pattern=re.compile(r"\bfind\b.*-delete\b"),
        category=CATEGORY_DESTRUCTIVE_FIND,
        description="Destructive find with -delete",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bfind\b.*-exec\s+rm\b"),
        category=CATEGORY_DESTRUCTIVE_FIND,
        description="Destructive find with -exec rm",
        severity="critical",
    ),
    # Destructive git
    DangerousPattern(
        pattern=re.compile(r"\bgit\s+push\s+--force\b"),
        category=CATEGORY_DESTRUCTIVE_GIT,
        description="Force push to git remote",
        severity="warning",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bgit\s+reset\s+--hard\b"),
        category=CATEGORY_DESTRUCTIVE_GIT,
        description="Hard reset of git working tree",
        severity="warning",
    ),
    # Device writes
    DangerousPattern(
        pattern=re.compile(r">\s*/dev/sd[a-z]"),
        category=CATEGORY_DEVICE_WRITE,
        description="Direct write to block device",
        severity="critical",
    ),
    DangerousPattern(
        pattern=re.compile(r"\bdd\s+.*of=/dev/"),
        category=CATEGORY_DEVICE_WRITE,
        description="dd write to device (dd of=/dev/)",
        severity="critical",
    ),
]


def check_command(command: str) -> list[DangerousPattern]:
    """Return all dangerous patterns that match the given command."""
    matches: list[DangerousPattern] = []
    for dp in DANGEROUS_PATTERNS:
        if dp.pattern.search(command):
            matches.append(dp)
    return matches


def is_dangerous(command: str) -> bool:
    """Return True if any dangerous pattern matches the command."""
    for dp in DANGEROUS_PATTERNS:
        if dp.pattern.search(command):
            return True
    return False


def format_warning(command: str, matches: list[DangerousPattern]) -> str:
    """Format a human-readable warning message for detected dangerous patterns."""
    if not matches:
        return "No dangerous patterns detected."

    lines = [f"WARNING: Dangerous command detected in: {command}", ""]
    for match in matches:
        severity_label = match.severity.upper()
        lines.append(f"  [{severity_label}] {match.description} (category: {match.category})")

    critical_count = sum(1 for m in matches if m.severity == "critical")
    warning_count = sum(1 for m in matches if m.severity == "warning")
    lines.append("")
    lines.append(f"Total: {critical_count} critical, {warning_count} warning")
    return "\n".join(lines)
