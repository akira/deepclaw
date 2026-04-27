"""Safety checks for DeepClaw: dangerous command detection, path validation,
credential redaction, and SSRF protection.

Provides regex-based pattern matching to identify dangerous shell commands,
destructive SQL statements, and other risky operations before they execute.
Also provides SSRF protection via URL validation that blocks requests to
private/internal networks, a write deny list for sensitive filesystem paths,
and credential leak redaction for tool output.
"""

import asyncio
import ipaddress
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


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
CATEGORY_PRIVILEGE_ESCALATION = "privilege_escalation"
CATEGORY_PACKAGE_MANAGEMENT = "package_management"
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
        pattern=re.compile(r"\bsudo\b"),
        category=CATEGORY_PRIVILEGE_ESCALATION,
        description="Privilege escalation via sudo",
        severity="warning",
    ),
    DangerousPattern(
        pattern=re.compile(
            r"\b(apt|apt-get)\s+(update|upgrade|install|remove|purge|dist-upgrade)\b"
        ),
        category=CATEGORY_PACKAGE_MANAGEMENT,
        description="System package management command",
        severity="warning",
    ),
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
        pattern=re.compile(r"\bpython(?:3)?\s+-?\s*<<"),
        category=CATEGORY_CODE_INJECTION,
        description="Inline Python heredoc execution (python <<EOF)",
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
    return any(dp.pattern.search(command) for dp in DANGEROUS_PATTERNS)


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


# ---------------------------------------------------------------------------
# Write path safety — deny list for sensitive filesystem paths
# ---------------------------------------------------------------------------

WRITE_DENIED_PREFIXES: list[str] = [
    "~/.ssh/",
    "~/.aws/",
    "~/.gnupg/",
    "~/.kube/",
    "/etc/sudoers.d/",
    "/etc/systemd/",
]

WRITE_DENIED_PATHS: list[str] = [
    "~/.ssh/id_rsa",
    "~/.ssh/id_ed25519",
    "~/.ssh/authorized_keys",
    "~/.ssh/config",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
    "~/.bash_profile",
    "~/.zprofile",
    "~/.npmrc",
    "~/.pypirc",
    "~/.pgpass",
    "~/.netrc",
    "~/.docker/config.json",
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/hosts",
]

_RESOLVED_DENIED_PREFIXES: list[str] | None = None
_RESOLVED_DENIED_PATHS: set[str] | None = None


def _resolve_denied_paths() -> tuple[list[str], set[str]]:
    """Lazily resolve ~ in deny lists. Cached after first call."""
    global _RESOLVED_DENIED_PREFIXES, _RESOLVED_DENIED_PATHS  # noqa: PLW0603
    if _RESOLVED_DENIED_PREFIXES is None:
        _RESOLVED_DENIED_PREFIXES = [os.path.expanduser(p) for p in WRITE_DENIED_PREFIXES]
        _RESOLVED_DENIED_PATHS = {os.path.expanduser(p) for p in WRITE_DENIED_PATHS}
    return _RESOLVED_DENIED_PREFIXES, _RESOLVED_DENIED_PATHS


def check_write_path(path: str) -> tuple[bool, str]:
    """Check whether a file path is safe to write to.

    Returns (is_safe, reason). Blocks writes to sensitive system and credential paths.
    """
    resolved = os.path.normpath(os.path.expanduser(path))
    prefixes, exact_paths = _resolve_denied_paths()

    if resolved in exact_paths:
        return (False, f"Write denied: {path} is a protected path")

    for prefix in prefixes:
        if resolved.startswith(prefix):
            return (False, f"Write denied: {path} is under protected directory {prefix}")

    return (True, "")


# ---------------------------------------------------------------------------
# Credential leak redaction
# ---------------------------------------------------------------------------

SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ghp_[A-Za-z0-9_]{36,}"),  # GitHub PAT
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),  # GitHub fine-grained PAT
    re.compile(r"gho_[A-Za-z0-9_]{36,}"),  # GitHub OAuth token
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI / Anthropic key
    re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),  # Anthropic API key
    re.compile(r"AKIA[A-Z0-9]{16}"),  # AWS access key ID
    re.compile(r"xoxb-[A-Za-z0-9\-]{20,}"),  # Slack bot token
    re.compile(r"xoxp-[A-Za-z0-9\-]{20,}"),  # Slack user token
    re.compile(r"glpat-[A-Za-z0-9\-_]{20,}"),  # GitLab PAT
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}=*"),  # Bearer token
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[=:]\s*['\"]?[A-Za-z0-9\-._~+/]{16,}"),
]

_REDACTED = "[REDACTED]"

# Explicit DeepClaw-managed/tool-managed vars blocked from child shell commands
# unless a trusted skill or config entry explicitly allowlists them.
CHILD_PROCESS_ENV_BLOCKLIST: frozenset[str] = frozenset(
    {
        "TELEGRAM_BOT_TOKEN",
        "DEEPCLAW_ALLOWED_USERS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
    }
)


def redact_secrets(text: str) -> str:
    """Replace detected credential patterns in text with [REDACTED]."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def has_secrets(text: str) -> bool:
    """Return True if any credential pattern is found in text."""
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


# ---------------------------------------------------------------------------
# Environment variable scrubbing
# ---------------------------------------------------------------------------

SAFE_ENV_VARS: frozenset[str] = frozenset(
    {
        # System essentials
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TERM_PROGRAM",
        "TMPDIR",
        "PWD",
        "OLDPWD",
        "HOSTNAME",
        "SHLVL",
        # Locale
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LC_COLLATE",
        "LC_TIME",
        "LC_NUMERIC",
        "LC_MONETARY",
        # XDG directories
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
        # Editor / pager
        "EDITOR",
        "VISUAL",
        "PAGER",
        "LESS",
        "LESSOPEN",
        # Toolchains — safe path/config vars, no secrets
        "CARGO_HOME",
        "RUSTUP_HOME",
        "GOPATH",
        "GOROOT",
        "GOBIN",
        "NODE_PATH",
        "NVM_DIR",
        "PYENV_ROOT",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "JAVA_HOME",
        "SDKMAN_DIR",
        "RBENV_ROOT",
        "GEM_HOME",
        "GEM_PATH",
        "BUNDLE_PATH",
        # Build / CI metadata (non-secret)
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_REPOSITORY",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_RUN_ID",
        # Misc safe vars
        "COLORTERM",
        "CLICOLOR",
        "FORCE_COLOR",
        "NO_COLOR",
        "COLUMNS",
        "LINES",
        "TZ",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "SSH_AUTH_SOCK",
        "GPG_TTY",
        "MANPATH",
        "INFOPATH",
    }
)

_SENSITIVE_SUBSTRINGS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "PRIVATE",
)


def is_sensitive_env_var_name(name: str) -> bool:
    """Return True when an env-var name would be scrubbed as sensitive by default."""
    upper = str(name).upper()
    return upper not in SAFE_ENV_VARS and any(s in upper for s in _SENSITIVE_SUBSTRINGS)


def scrub_env(
    env: dict[str, str] | None = None, *, keep_sensitive: set[str] | frozenset[str] | None = None
) -> dict[str, str]:
    """Return a filtered copy of environment variables safe for generic child processes.

    This is the conservative scrubber used when DeepClaw wants a mostly non-secret
    subprocess environment. It strips any variable whose name looks sensitive
    unless explicitly allowlisted via ``keep_sensitive``.
    """
    source = env if env is not None else os.environ
    allowed_sensitive = set(keep_sensitive or ())
    filtered: dict[str, str] = {}
    for name, value in source.items():
        if name in SAFE_ENV_VARS or name in allowed_sensitive:
            filtered[name] = value
            continue
        if is_sensitive_env_var_name(name):
            continue
        filtered[name] = value
    return filtered


def sanitize_child_command_env(
    env: dict[str, str] | None = None,
    extra_env: dict[str, str] | None = None,
    *,
    keep_sensitive: set[str] | frozenset[str] | None = None,
) -> dict[str, str]:
    """Build the env for DeepClaw child shell commands.

    Start from the conservative ``scrub_env()`` baseline, then allowlist any
    explicitly approved sensitive variables. This preserves the pre-existing
    default of not forwarding arbitrary secret-looking env vars into child
    shell commands while still allowing narrow opt-ins.
    """
    merged = dict(env if env is not None else os.environ)
    merged.update(extra_env or {})
    return scrub_env(merged, keep_sensitive=keep_sensitive)


# ---------------------------------------------------------------------------
# SSRF Protection — URL safety validation
# ---------------------------------------------------------------------------

BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    # RFC 1918 private ranges
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    # Loopback
    ipaddress.IPv4Network("127.0.0.0/8"),
    # Link-local
    ipaddress.IPv4Network("169.254.0.0/16"),
    # Unspecified
    ipaddress.IPv4Network("0.0.0.0/8"),
    # CGNAT / RFC 6598
    ipaddress.IPv4Network("100.64.0.0/10"),
    # Multicast
    ipaddress.IPv4Network("224.0.0.0/4"),
    # Reserved
    ipaddress.IPv4Network("240.0.0.0/4"),
    # IPv6 loopback
    ipaddress.IPv6Network("::1/128"),
    # IPv6 unique local
    ipaddress.IPv6Network("fc00::/7"),
    # IPv6 link-local
    ipaddress.IPv6Network("fe80::/10"),
]

BLOCKED_HOSTNAMES: set[str] = {
    "metadata.google.internal",
    "metadata.goog",
    "169.254.169.254",
}

_REASON_INVALID_URL = "Invalid URL"
_REASON_DNS_FAILED = "DNS resolution failed"


def _is_ip_blocked(ip_str: str) -> bool:
    """Check whether an IP address falls within any blocked network."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in network for network in BLOCKED_NETWORKS)


def _extract_hostname(url: str) -> str | None:
    """Extract and return the hostname from a URL, or None if invalid."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if not parsed.scheme or not parsed.hostname:
        return None
    return parsed.hostname


def check_url_safety_sync(url: str) -> tuple[bool, str]:
    """Synchronous URL safety check — blocks requests to private/internal networks.

    Returns (is_safe, reason). Fail-closed: DNS failures return unsafe.
    """
    hostname = _extract_hostname(url)
    if hostname is None:
        return (False, _REASON_INVALID_URL)

    if hostname in BLOCKED_HOSTNAMES:
        return (False, f"Blocked hostname: {hostname}")

    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, OSError):
        return (False, _REASON_DNS_FAILED)

    for addrinfo in addrinfos:
        ip_str = addrinfo[4][0]
        if _is_ip_blocked(ip_str):
            return (False, f"IP {ip_str} is in a blocked network range")

    return (True, "")


async def check_url_safety(url: str) -> tuple[bool, str]:
    """Async URL safety check — blocks requests to private/internal networks.

    Returns (is_safe, reason). Fail-closed: DNS failures return unsafe.
    """
    hostname = _extract_hostname(url)
    if hostname is None:
        return (False, _REASON_INVALID_URL)

    if hostname in BLOCKED_HOSTNAMES:
        return (False, f"Blocked hostname: {hostname}")

    try:
        addrinfos = await asyncio.to_thread(socket.getaddrinfo, hostname, None)
    except (socket.gaierror, OSError):
        return (False, _REASON_DNS_FAILED)

    for addrinfo in addrinfos:
        ip_str = addrinfo[4][0]
        if _is_ip_blocked(ip_str):
            return (False, f"IP {ip_str} is in a blocked network range")

    return (True, "")
