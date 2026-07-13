"""Security helpers for input validation, path containment, and output escaping."""

from __future__ import annotations

import html
import ipaddress
import shlex
import subprocess  # nosec B404
from pathlib import Path
from urllib.parse import urlparse


class SecurityError(ValueError):
    """Raised when a security check fails."""


# Cloud metadata and other sensitive endpoints that should never be allowed.
_BLOCKED_HOSTS: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.aws.internal",
        "metadata.azure.internal",
    }
)

# Hosts that are allowed by default for local benchmarking.
_DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",
    }
)

# IP networks considered private, link-local, or loopback.
_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _is_private_host(host: str) -> bool:
    """Return True if *host* resolves to a private/reserved IP address."""
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _PRIVATE_NETWORKS)
    except ValueError:
        # Not an IP literal; rely on host-name checks elsewhere.
        return False


def validate_base_url(
    url: str,
    *,
    allow_internal: bool = False,
    extra_allowed_hosts: set[str] | frozenset[str] | None = None,
) -> str:
    """Validate that *url* is an acceptable OpenAI-compatible base URL.

    By default only ``localhost`` and loopback addresses are permitted because
    ``toks-bench`` is designed for benchmarking local inference servers. Set
    ``allow_internal=True`` to additionally allow RFC1918/private IPs and
    non-loopback link-local addresses (useful for LAN benchmarking). Cloud
    metadata endpoints are always blocked.

    Raises:
        SecurityError: If the URL scheme is not ``http``/``https``, the host is
            blocked, or the URL contains a username/password.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SecurityError(f"base_url must use http:// or https://: {url}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise SecurityError(f"base_url must contain a host: {url}")

    if parsed.username is not None or parsed.password is not None:
        raise SecurityError(f"base_url must not contain credentials: {url}")

    if host in _BLOCKED_HOSTS:
        raise SecurityError(f"base_url host is blocked: {host}")

    allowed_hosts = set(_DEFAULT_ALLOWED_HOSTS)
    if extra_allowed_hosts:
        allowed_hosts.update(extra_allowed_hosts)
    if host in allowed_hosts:
        return url

    if allow_internal:
        if _is_private_host(host):
            return url
        # Allow hostnames that resolve to private addresses?  We only inspect the
        # literal here; DNS resolution is intentionally not performed to avoid SSRF.
        raise SecurityError(
            f"base_url host not in allowlist and does not appear to be private: {url}"
        )

    raise SecurityError(
        f"base_url host not in allowlist (use --allow-internal-urls to permit private hosts): {url}"
    )


def resolve_contained_path(
    raw_path: str | Path,
    workspace: Path,
    *,
    must_exist: bool = False,
    follow_symlinks: bool = True,
) -> Path:
    """Resolve *raw_path* and ensure it is contained within *workspace*.

    The workspace is expanded to its absolute, real (symlink-resolved) path.
    The input is rejected if any component is ``..`` after normalisation, if it
    is an absolute path outside the workspace, or if symlink resolution escapes
    the workspace.

    Raises:
        SecurityError: If the path escapes the workspace.
        FileNotFoundError: If *must_exist* is True and the path does not exist.
    """
    workspace = workspace.resolve(strict=False)
    raw = Path(raw_path)

    if raw.is_absolute():
        # Absolute paths are only safe if they point inside the workspace.
        candidate = raw.resolve(strict=False)
    else:
        candidate = (workspace / raw).resolve(strict=False)

    # Reject paths that contain a '..' component after normalisation.  Using
    # Path.resolve() collapses them, but we also explicitly check the original
    # string form as defense-in-depth.
    parts = raw.parts
    if ".." in parts:
        raise SecurityError(f"path contains '..' and is not allowed: {raw_path}")

    # For non-existent intermediate components resolve() does not follow
    # symlinks.  When the path exists, perform an extra realpath check to catch
    # symlink escapes.
    if follow_symlinks and candidate.exists():
        real_candidate = candidate.resolve(strict=True)
        real_workspace = workspace.resolve(strict=True)
        try:
            real_candidate.relative_to(real_workspace)
        except ValueError as exc:
            raise SecurityError(
                f"resolved path escapes workspace via symlink: {raw_path}"
            ) from exc

    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise SecurityError(f"path escapes workspace: {raw_path}") from exc

    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"path does not exist: {candidate}")

    return candidate


def run_sandboxed(
    cmd: list[str],
    *,
    cwd: Path | str | None = None,
    check: bool = True,
    capture_output: bool = False,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command without invoking a shell.

    The command must be passed as a sequence of arguments.  Shell metacharacters
    in the arguments are treated literally, not interpreted.  This is the safe
    replacement for ``subprocess.run(..., shell=True)``.
    """
    if not cmd:
        raise SecurityError("command list must not be empty")
    if not all(isinstance(arg, str) for arg in cmd):
        raise SecurityError("all command arguments must be strings")
    if any(arg is None for arg in cmd):
        raise SecurityError("command arguments must not be None")

    return subprocess.run(  # noqa: S603  # nosec B603
        cmd,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
        env=env,
        shell=False,
    )


def sanitize_csv_field(value: str) -> str:
    """Neutralise CSV formula injection.

    Values beginning with characters that spreadsheet applications interpret as
    formulas (``=``, ``+``, ``-``, ``@``, tab) are prefixed with a single quote.
    This prevents CSV injection (e.g. ``=cmd|'/C calc'!A0``) when the file is
    opened in Excel/LibreOffice Calc.
    """
    if not isinstance(value, str):
        value = str(value)
    if value and value[0] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def escape_html(value: str) -> str:
    """Escape a string for safe insertion into HTML text or attribute context."""
    return html.escape(str(value), quote=True)


def escape_markdown(value: str) -> str:
    """Escape characters that have special meaning in Markdown inline text.

    This is a lightweight escaping suitable for table cells and plain text.  It
    does not attempt to sanitise raw HTML; use ``escape_html`` for HTML output.
    """
    value = str(value)
    # Escape backslash first to avoid double-escaping.
    value = value.replace("\\", "\\\\")
    # Escape inline markdown metacharacters.
    for ch in ("*", "_", "`", "[", "]", "<", ">", "&"):
        value = value.replace(ch, "\\" + ch)
    return value


def validate_command_no_shell(command: str) -> list[str]:
    """Split a command string safely using POSIX rules.

    This is a convenience wrapper around :func:`shlex.split` for the rare cases
    where a user-supplied string must be turned into an argument list.  It
    refuses strings that contain shell control operators.
    """
    if ";" in command or "|" in command or "&" in command or "$" in command:
        raise SecurityError(f"command contains disallowed shell metacharacters: {command}")
    return shlex.split(command)
