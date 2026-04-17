"""Tests for providers/claude_cli.py (subprocess mocked)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pipeline_youtube.providers.claude_cli import (
    ClaudeCliError,
    ClaudeResponse,
    invoke_claude,
)

# =====================================================
# Helpers
# =====================================================


def _completed(returncode: int = 0, stdout: str = "{}", stderr: str = "") -> MagicMock:
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.returncode = returncode
    mock.stdout = stdout
    mock.stderr = stderr
    return mock


SAMPLE_RESPONSE = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "duration_ms": 1838,
    "duration_api_ms": 1823,
    "num_turns": 1,
    "result": "Hey",
    "stop_reason": "end_turn",
    "session_id": "ccbefeda-2138-429c-a99d-9792e71f64a4",
    "total_cost_usd": 0.02982025,
    "usage": {
        "input_tokens": 9,
        "output_tokens": 86,
        "cache_creation_input_tokens": 23505,
        "cache_read_input_tokens": 0,
    },
    "modelUsage": {
        "claude-haiku-4-5-20251001": {
            "inputTokens": 9,
            "outputTokens": 86,
        }
    },
}


# =====================================================
# Successful invocation
# =====================================================


class TestSuccessfulInvocation:
    def test_parses_standard_json_response(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            resp = invoke_claude("hi", model="haiku")

        assert isinstance(resp, ClaudeResponse)
        assert resp.text == "Hey"
        assert resp.model == "haiku"
        assert resp.input_tokens == 9
        assert resp.output_tokens == 86
        assert resp.cache_creation_tokens == 23505
        assert resp.cache_read_tokens == 0
        assert resp.total_cost_usd == 0.02982025
        assert resp.duration_ms == 1838
        assert resp.session_id == "ccbefeda-2138-429c-a99d-9792e71f64a4"
        assert resp.stop_reason == "end_turn"

    def test_total_tokens_property(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            resp = invoke_claude("hi", model="haiku")

        # input + cache_creation + output = 9 + 23505 + 86 = 23600
        assert resp.total_tokens == 23600

    def test_prompt_passed_via_stdin_not_positional(self):
        """Long prompts must go via stdin to avoid --tools variadic parser eating them."""
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("my prompt text", model="sonnet")

        call_args = run.call_args
        cmd = call_args.args[0]
        assert cmd[0] == "claude"
        assert cmd[1] == "-p"
        # Prompt must NOT appear as positional in cmd
        assert "my prompt text" not in cmd
        # Prompt must be piped via stdin
        assert call_args.kwargs.get("input") == "my prompt text"

    def test_model_flag_passed(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", model="sonnet")

        cmd = run.call_args.args[0]
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "sonnet"

    def test_output_format_is_json(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi")

        cmd = run.call_args.args[0]
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"


# =====================================================
# Flag composition
# =====================================================


class TestFlagComposition:
    def test_disallow_tools_default_true(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi")

        cmd = run.call_args.args[0]
        assert "--tools" in cmd
        # --tools "" must be the LAST value-taking flag to avoid variadic
        # argument parser consuming subsequent args.
        tools_idx = cmd.index("--tools")
        assert cmd[tools_idx + 1] == ""

    def test_disallow_tools_false_omits_flag(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", disallow_tools=False)

        cmd = run.call_args.args[0]
        assert "--tools" not in cmd

    def test_tools_flag_goes_last(self):
        """--tools must come after all other value-taking flags."""
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude(
                "hi",
                append_system_prompt="role prompt",
                resume_session="abc-123",
            )

        cmd = run.call_args.args[0]
        assert "--tools" in cmd
        tools_idx = cmd.index("--tools")
        # No --append-system-prompt or --resume after --tools
        assert "--append-system-prompt" not in cmd[tools_idx:]
        assert "--resume" not in cmd[tools_idx:]

    def test_append_system_prompt(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", append_system_prompt="You are topic extractor.")

        cmd = run.call_args.args[0]
        assert "--append-system-prompt" in cmd
        assert cmd[cmd.index("--append-system-prompt") + 1] == "You are topic extractor."

    def test_system_prompt_replace(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", system_prompt="clean role-only prompt")

        cmd = run.call_args.args[0]
        assert "--system-prompt" in cmd
        assert "--append-system-prompt" not in cmd
        assert cmd[cmd.index("--system-prompt") + 1] == "clean role-only prompt"

    def test_system_and_append_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            invoke_claude("hi", system_prompt="a", append_system_prompt="b")

    def test_no_session_persistence_default(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi")

        cmd = run.call_args.args[0]
        assert "--no-session-persistence" in cmd

    def test_persist_session_true_omits_flag(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", persist_session=True)

        cmd = run.call_args.args[0]
        assert "--no-session-persistence" not in cmd

    def test_resume_session(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", resume_session="session-xyz")

        cmd = run.call_args.args[0]
        assert "--resume" in cmd
        assert cmd[cmd.index("--resume") + 1] == "session-xyz"

    def test_max_budget_usd(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", max_budget_usd=5.0)

        cmd = run.call_args.args[0]
        assert "--max-budget-usd" in cmd
        assert cmd[cmd.index("--max-budget-usd") + 1] == "5.0"

    def test_extra_args_appended(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(SAMPLE_RESPONSE))
            invoke_claude("hi", extra_args=["--fallback-model", "haiku"])

        cmd = run.call_args.args[0]
        assert "--fallback-model" in cmd
        assert cmd[cmd.index("--fallback-model") + 1] == "haiku"


# =====================================================
# Error handling
# =====================================================


class TestErrorHandling:
    def test_nonzero_exit_raises(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(returncode=1, stdout="", stderr="oops something broke")
            with pytest.raises(ClaudeCliError, match="exited 1"):
                invoke_claude("hi")

    def test_invalid_json_raises(self):
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout="not valid json at all")
            with pytest.raises(ClaudeCliError, match="invalid JSON"):
                invoke_claude("hi")

    def test_is_error_true_raises(self):
        payload = dict(SAMPLE_RESPONSE)
        payload["is_error"] = True
        payload["subtype"] = "error"
        payload["result"] = "something went wrong"
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(payload))
            with pytest.raises(ClaudeCliError, match="is_error=true"):
                invoke_claude("hi")

    def test_timeout_raises(self):
        with patch("subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=30)
            with pytest.raises(ClaudeCliError, match="timeout"):
                invoke_claude("hi", timeout=30)

    def test_file_not_found_raises(self):
        with patch("subprocess.run") as run:
            run.side_effect = FileNotFoundError("claude binary missing")
            with pytest.raises(ClaudeCliError, match="`claude` CLI not found"):
                invoke_claude("hi")

    def test_missing_usage_fields_graceful(self):
        """If usage dict is partial, None should be returned — not KeyError."""
        minimal = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "ok",
            "session_id": "s1",
            "usage": {"input_tokens": 5},  # no output_tokens, no cache fields
        }
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(minimal))
            resp = invoke_claude("hi")

        assert resp.text == "ok"
        assert resp.input_tokens == 5
        assert resp.output_tokens is None
        assert resp.cache_creation_tokens is None

    def test_empty_result_ok(self):
        """Empty string result is valid (e.g. model refused to generate)."""
        payload = dict(SAMPLE_RESPONSE)
        payload["result"] = ""
        with patch("subprocess.run") as run:
            run.return_value = _completed(stdout=json.dumps(payload))
            resp = invoke_claude("hi")

        assert resp.text == ""
