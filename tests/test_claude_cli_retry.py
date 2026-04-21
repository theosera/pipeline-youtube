"""Tests for the retry-on-transient-error logic in `invoke_claude`.

These tests mock `subprocess.run` to simulate transient failures and
verify that:

1. Known transient error patterns trigger a retry.
2. Non-transient errors propagate immediately (no retry).
3. Retry gives up after `max_retries` and raises the last error.
4. `max_retries=0` disables retry behavior (one attempt only).
5. Exponential backoff is applied between retries.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pipeline_youtube.providers import claude_cli as claude_cli_mod
from pipeline_youtube.providers.claude_cli import (
    ClaudeCliError,
    _is_transient_error_text,
    invoke_claude,
)

_FAKE_CLAUDE_BIN = "/opt/test/bin/claude"


@pytest.fixture(autouse=True)
def _seed_resolved_claude_bin(monkeypatch):
    monkeypatch.setattr(claude_cli_mod, "_CLAUDE_BIN", _FAKE_CLAUDE_BIN)
    monkeypatch.setattr(claude_cli_mod, "_CLAUDE_VERSION", "claude 2.1.109")
    yield
    claude_cli_mod._reset_claude_binary_cache_for_tests()


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Replace time.sleep with a no-op so retry tests run instantly."""
    monkeypatch.setattr(claude_cli_mod.time, "sleep", lambda _s: None)


def _completed(returncode: int = 0, stdout: str = "{}", stderr: str = "") -> MagicMock:
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


_SUCCESS = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "ok",
    "session_id": "s1",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _transient_payload(
    msg: str = "API Error: Stream idle timeout - partial response received",
) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": msg,
            "session_id": "s_err",
        }
    )


# =====================================================
# Pattern matcher
# =====================================================


class TestTransientPatternMatcher:
    @pytest.mark.parametrize(
        "text",
        [
            "API Error: Stream idle timeout - partial response received",
            "API Error: Unable to connect to API (ConnectionRefused)",
            "API Error: Unable to connect to API (FailedToOpenSocket)",
            "network: ECONNRESET",
            "socket hang up",
            "503 Service Unavailable",
            "502 Bad Gateway",
            "overloaded_error",
        ],
    )
    def test_known_transient_matches(self, text: str):
        assert _is_transient_error_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            "authentication failed",
            "invalid API key",
            "context length exceeded",
            "",
        ],
    )
    def test_non_transient_does_not_match(self, text: str):
        assert not _is_transient_error_text(text)


# =====================================================
# Retry behavior
# =====================================================


class TestRetryBehavior:
    def test_transient_error_then_success(self):
        """First call hits stream-idle-timeout, second succeeds."""
        with patch("subprocess.run") as run:
            run.side_effect = [
                _completed(returncode=1, stdout=_transient_payload()),
                _completed(returncode=0, stdout=json.dumps(_SUCCESS)),
            ]
            resp = invoke_claude("hi", retry_base_delay=0.0)

        assert resp.text == "ok"
        assert run.call_count == 2

    def test_two_transient_errors_then_success(self):
        with patch("subprocess.run") as run:
            run.side_effect = [
                _completed(returncode=1, stdout=_transient_payload("ConnectionRefused")),
                _completed(returncode=1, stdout=_transient_payload("Stream idle timeout")),
                _completed(returncode=0, stdout=json.dumps(_SUCCESS)),
            ]
            resp = invoke_claude("hi", retry_base_delay=0.0)

        assert resp.text == "ok"
        assert run.call_count == 3

    def test_exhaust_retries_raises_final_error(self):
        """After max_retries transient failures, raise the last one."""
        with patch("subprocess.run") as run:
            run.return_value = _completed(returncode=1, stdout=_transient_payload())
            with pytest.raises(ClaudeCliError, match="Stream idle timeout"):
                invoke_claude("hi", max_retries=2, retry_base_delay=0.0)

        # 1 initial + 2 retries = 3 attempts
        assert run.call_count == 3

    def test_max_retries_zero_disables_retry(self):
        """max_retries=0 means one attempt, never retry."""
        with patch("subprocess.run") as run:
            run.return_value = _completed(returncode=1, stdout=_transient_payload())
            with pytest.raises(ClaudeCliError):
                invoke_claude("hi", max_retries=0, retry_base_delay=0.0)

        assert run.call_count == 1

    def test_non_transient_error_propagates_immediately(self):
        """Auth error / parse error etc. should NOT retry."""
        payload = json.dumps(
            {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "result": "authentication failed",
                "session_id": "s_err",
            }
        )
        with patch("subprocess.run") as run:
            run.return_value = _completed(returncode=1, stdout=payload)
            with pytest.raises(ClaudeCliError, match="authentication failed"):
                invoke_claude("hi", max_retries=3, retry_base_delay=0.0)

        assert run.call_count == 1

    def test_subprocess_timeout_does_not_retry(self):
        """subprocess.TimeoutExpired is permanent — we already waited the full budget."""
        with patch("subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=30)
            with pytest.raises(ClaudeCliError, match="timeout"):
                invoke_claude("hi", timeout=30, max_retries=3, retry_base_delay=0.0)

        assert run.call_count == 1

    def test_invalid_json_does_not_retry(self):
        """A broken JSON response is not a network issue — don't retry."""
        with patch("subprocess.run") as run:
            run.return_value = _completed(returncode=0, stdout="definitely not json")
            with pytest.raises(ClaudeCliError, match="invalid JSON"):
                invoke_claude("hi", max_retries=3, retry_base_delay=0.0)

        assert run.call_count == 1

    def test_exponential_backoff_between_retries(self):
        """Sleep delays should double: 30 → 60 → 120."""
        with (
            patch("subprocess.run") as run,
            patch.object(claude_cli_mod.time, "sleep") as sleep_mock,
        ):
            run.return_value = _completed(returncode=1, stdout=_transient_payload())
            with pytest.raises(ClaudeCliError):
                invoke_claude("hi", max_retries=3, retry_base_delay=30.0)

        # 3 retries → 3 sleep calls
        delays = [call.args[0] for call in sleep_mock.call_args_list]
        assert delays == [30.0, 60.0, 120.0]

    def test_retry_base_delay_override(self):
        """Custom base delay should propagate through exponential formula."""
        with (
            patch("subprocess.run") as run,
            patch.object(claude_cli_mod.time, "sleep") as sleep_mock,
        ):
            run.return_value = _completed(returncode=1, stdout=_transient_payload())
            with pytest.raises(ClaudeCliError):
                invoke_claude("hi", max_retries=2, retry_base_delay=5.0)

        delays = [call.args[0] for call in sleep_mock.call_args_list]
        assert delays == [5.0, 10.0]


# =====================================================
# Transient flag on ClaudeCliError
# =====================================================


class TestTransientFlag:
    def test_transient_flag_set_on_stream_idle(self):
        """_invoke_claude_once should mark transient=True for known patterns."""
        with patch("subprocess.run") as run:
            run.return_value = _completed(returncode=1, stdout=_transient_payload())
            # Use max_retries=0 so we see the bare exception
            with pytest.raises(ClaudeCliError) as exc_info:
                invoke_claude("hi", max_retries=0)

        assert exc_info.value.transient is True

    def test_transient_flag_not_set_on_auth_error(self):
        payload = json.dumps(
            {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "result": "authentication failed: bad token",
                "session_id": "s_err",
            }
        )
        with patch("subprocess.run") as run:
            run.return_value = _completed(returncode=1, stdout=payload)
            with pytest.raises(ClaudeCliError) as exc_info:
                invoke_claude("hi", max_retries=0)

        assert exc_info.value.transient is False

    def test_transient_flag_not_set_on_timeout(self):
        with patch("subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=30)
            with pytest.raises(ClaudeCliError) as exc_info:
                invoke_claude("hi", timeout=30, max_retries=0)

        assert exc_info.value.transient is False
