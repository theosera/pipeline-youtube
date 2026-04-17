"""Wrapper around `claude -p` headless CLI.

All AI calls in the pipeline go through this module. It spawns
`claude -p` as a subprocess so the user's Claude Pro/Max OAuth session
is used automatically — no ANTHROPIC_API_KEY required.

Prompt passing strategy
-----------------------
The prompt is piped via **stdin**, not the positional argument. Reason:
several `claude -p` flags use variadic parsers (e.g. `--tools <tools...>`)
that would otherwise consume the following positional prompt. Piping via
stdin avoids argument-parsing ambiguity entirely and also sidesteps the
`ARG_MAX` limit for very large Agent Teams prompts.

JSON output schema (verified at runtime on claude-code 2.1.109)
---------------------------------------------------------------
```
{
  "type": "result",
  "subtype": "success" | "error",
  "is_error": bool,
  "duration_ms": int,
  "duration_api_ms": int,
  "num_turns": int,
  "result": "...model text...",
  "session_id": "uuid",
  "total_cost_usd": float,       # theoretical API cost; $0 out-of-pocket on OAuth
  "stop_reason": "end_turn" | ...,
  "usage": {
    "input_tokens": int,
    "output_tokens": int,
    "cache_creation_input_tokens": int,
    "cache_read_input_tokens": int,
    ...
  },
  "modelUsage": { "<full_model_id>": {...} },
}
```
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClaudeResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    session_id: str | None = None
    stop_reason: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """Sum of fresh input + cache creation + output (cache reads are ~free)."""
        return (
            (self.input_tokens or 0) + (self.cache_creation_tokens or 0) + (self.output_tokens or 0)
        )


class ClaudeCliError(RuntimeError):
    """Raised when `claude -p` exits non-zero or returns unparseable JSON."""


def invoke_claude(
    prompt: str,
    *,
    append_system_prompt: str | None = None,
    system_prompt: str | None = None,
    model: str = "sonnet",
    disallow_tools: bool = True,
    timeout: int = 600,
    resume_session: str | None = None,
    persist_session: bool = False,
    max_budget_usd: float | None = None,
    extra_args: list[str] | None = None,
) -> ClaudeResponse:
    """Call `claude -p` headless CLI and return a structured response.

    Parameters
    ----------
    prompt:
        The user message. Passed via stdin (not positional) to avoid
        argument-parser conflicts with variadic flags like `--tools`.
    append_system_prompt:
        Text to append to Claude Code's default system prompt. Use this
        for role-specific instructions that should not clobber defaults.
        Mutually exclusive with `system_prompt`.
    system_prompt:
        Replaces Claude Code's default system prompt entirely. Use this
        for Agent Teams where strict role isolation matters and default
        context (CLAUDE.md, memory, etc.) should be dropped. Cuts
        cache-creation overhead by ~23k tokens.
    model:
        Alias (`sonnet`, `haiku`, `opus`) or full model ID.
    disallow_tools:
        If True (default), passes `--tools ""` so Claude cannot invoke
        any built-in tools. Agent Teams roles are pure text generation.
    timeout:
        Seconds before `subprocess.run` kills the child. Agent Teams
        large-context calls may need > 600s — adjust per stage.
    resume_session:
        Session UUID to resume (from a previous ClaudeResponse). Enables
        shared cache across sequential role calls in Agent Teams.
    persist_session:
        If False (default), passes `--no-session-persistence` so the
        session is not saved to `claude --resume` history. Keeps the
        user's interactive /resume list clean.
    max_budget_usd:
        Optional cost cap (only applies to API-key users; OAuth users
        get a hard clamp from Anthropic's plan quotas).
    extra_args:
        Escape hatch for additional CLI flags not wrapped by this
        function. Example: `["--fallback-model", "haiku"]`.

    Returns
    -------
    ClaudeResponse with text, model, token counts, cost, session id.

    Raises
    ------
    ClaudeCliError if the subprocess exits non-zero or returns invalid
    JSON. Timeouts are also wrapped in ClaudeCliError.
    """
    if system_prompt is not None and append_system_prompt is not None:
        raise ValueError("system_prompt and append_system_prompt are mutually exclusive")

    cmd: list[str] = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        model,
    ]

    if system_prompt is not None:
        cmd.extend(["--system-prompt", system_prompt])
    elif append_system_prompt is not None:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    if not persist_session:
        cmd.append("--no-session-persistence")

    if resume_session is not None:
        cmd.extend(["--resume", resume_session])

    if max_budget_usd is not None:
        cmd.extend(["--max-budget-usd", str(max_budget_usd)])

    if extra_args:
        cmd.extend(extra_args)

    # IMPORTANT: --tools "" must go LAST among value-taking flags so
    # its variadic parser has no following positional to accidentally
    # consume. (We pass prompt via stdin so there's no positional, but
    # we still order it last to be safe against future refactors.)
    if disallow_tools:
        cmd.extend(["--tools", ""])

    # Strip ANTHROPIC_API_KEY from the subprocess environment so
    # `claude -p` uses OAuth (Pro/Max plan quota) instead of API
    # credit billing. This prevents accidental API credit consumption.
    import os

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCliError(f"claude -p timeout after {timeout}s (cmd: {shlex.join(cmd)})") from e
    except FileNotFoundError as e:
        raise ClaudeCliError(
            "`claude` CLI not found in PATH. Install Claude Code or run `claude login`."
        ) from e

    if result.returncode != 0:
        raise ClaudeCliError(
            f"claude -p exited {result.returncode}: "
            f"stderr={result.stderr[:500]!r} stdout={result.stdout[:200]!r}"
        )

    try:
        data: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCliError(
            f"invalid JSON from claude -p: {e}; stdout={result.stdout[:500]!r}"
        ) from e

    if data.get("is_error"):
        raise ClaudeCliError(
            f"claude -p returned is_error=true: "
            f"subtype={data.get('subtype')!r} result={str(data.get('result'))[:500]!r}"
        )

    usage = data.get("usage") or {}

    return ClaudeResponse(
        text=str(data.get("result", "")),
        model=model,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_creation_tokens=usage.get("cache_creation_input_tokens"),
        total_cost_usd=data.get("total_cost_usd"),
        duration_ms=data.get("duration_ms"),
        session_id=data.get("session_id"),
        stop_reason=data.get("stop_reason"),
        raw=data,
    )
