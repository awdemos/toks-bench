"""Tests for security helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from toks_bench.security import (
    SecurityError,
    escape_html,
    escape_markdown,
    resolve_contained_path,
    run_sandboxed,
    sanitize_csv_field,
    validate_base_url,
    validate_command_no_shell,
)


class TestValidateBaseUrl:
    def test_allows_localhost_http(self) -> None:
        assert validate_base_url("http://localhost:8080/v1") == "http://localhost:8080/v1"

    def test_allows_loopback_ipv4(self) -> None:
        assert validate_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"

    def test_allows_loopback_ipv6(self) -> None:
        assert validate_base_url("http://[::1]:8000/v1") == "http://[::1]:8000/v1"

    def test_rejects_metadata_endpoint(self) -> None:
        with pytest.raises(SecurityError, match="host is blocked"):
            validate_base_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(SecurityError, match="must use http:// or https://"):
            validate_base_url("ftp://localhost:8080/v1")

    def test_rejects_private_ip_by_default(self) -> None:
        with pytest.raises(SecurityError, match="not in allowlist"):
            validate_base_url("http://192.168.1.50:8080/v1")

    def test_allows_private_ip_when_requested(self) -> None:
        assert (
            validate_base_url("http://192.168.1.50:8080/v1", allow_internal=True)
            == "http://192.168.1.50:8080/v1"
        )

    def test_rejects_cloud_metadata_even_when_internal_allowed(self) -> None:
        with pytest.raises(SecurityError, match="host is blocked"):
            validate_base_url("http://169.254.169.254/latest/", allow_internal=True)

    def test_rejects_url_with_credentials(self) -> None:
        with pytest.raises(SecurityError, match="must not contain credentials"):
            validate_base_url("http://user:pass@localhost:8080/v1")

    def test_extra_allowed_hosts(self) -> None:
        assert (
            validate_base_url(
                "http://my-server.local:8080/v1",
                extra_allowed_hosts={"my-server.local"},
            )
            == "http://my-server.local:8080/v1"
        )


class TestResolveContainedPath:
    def test_allows_relative_path_inside_workspace(self, tmp_path: Path) -> None:
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "hello.txt").write_text("hi", encoding="utf-8")
        result = resolve_contained_path("prompts/hello.txt", tmp_path, must_exist=True)
        assert result == tmp_path / "prompts" / "hello.txt"

    def test_rejects_path_with_dotdot(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError, match="contains '\\.\\.'"):
            resolve_contained_path("../etc/passwd", tmp_path)

    def test_rejects_absolute_path_outside_workspace(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError, match="escapes workspace"):
            resolve_contained_path("/etc/passwd", tmp_path)

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        (tmp_path / "safe").mkdir()
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        (tmp_path / "safe" / "link.txt").symlink_to(outside)
        with pytest.raises(SecurityError, match="escapes workspace via symlink"):
            resolve_contained_path("safe/link.txt", tmp_path, must_exist=True)

    def test_allows_absolute_path_inside_workspace(self, tmp_path: Path) -> None:
        file = tmp_path / "inside.txt"
        file.write_text("x", encoding="utf-8")
        result = resolve_contained_path(str(file), tmp_path, must_exist=True)
        assert result == file


class TestRunSandboxed:
    def test_runs_command_as_argument_list(self) -> None:
        result = run_sandboxed(["echo", "hello"], capture_output=True, check=True)
        assert result.stdout.strip() == "hello"

    def test_does_not_invoke_shell(self) -> None:
        # A shell metacharacter should be treated as a literal argument.
        result = run_sandboxed(["echo", "*"], capture_output=True, check=True)
        assert result.stdout.strip() == "*"

    def test_rejects_empty_command(self) -> None:
        with pytest.raises(SecurityError, match="must not be empty"):
            run_sandboxed([])

    def test_rejects_none_argument(self) -> None:
        with pytest.raises(SecurityError, match="all command arguments must be strings"):
            # type: ignore[list-item]
            run_sandboxed(["echo", None])

    def test_check_false_does_not_raise(self) -> None:
        result = run_sandboxed(["false"], check=False)
        assert result.returncode == 1


class TestReportSanitization:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("normal", "normal"),
            ("=cmd|'/C calc'!A0", "'=cmd|'/C calc'!A0"),
            ("+cmd", "'+cmd"),
            ("-cmd", "'-cmd"),
            ("@cmd", "'@cmd"),
            ("\tformula", "'\tformula"),
        ],
    )
    def test_sanitize_csv_field(self, value: str, expected: str) -> None:
        assert sanitize_csv_field(value) == expected

    def test_escape_html(self) -> None:
        assert (
            escape_html("<script>alert('xss')</script>")
            == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        )

    def test_escape_markdown(self) -> None:
        assert escape_markdown("**bold**") == "\\*\\*bold\\*\\*"


class TestValidateCommandNoShell:
    def test_splits_simple_command(self) -> None:
        assert validate_command_no_shell("echo hello world") == ["echo", "hello", "world"]

    def test_rejects_shell_metacharacters(self) -> None:
        with pytest.raises(SecurityError, match="disallowed shell metacharacters"):
            validate_command_no_shell("echo foo; rm -rf /")

        with pytest.raises(SecurityError, match="disallowed shell metacharacters"):
            validate_command_no_shell("cat file | bash")
