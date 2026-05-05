# CLAUDE.md — pipeline-youtube-repo

This is the authoritative CLAUDE.md for this repository. It is version-controlled;
any modification is visible in `git diff` and requires review.

## Project

YouTube playlist → Obsidian vault learning pipeline written in Python 3.13.

## Commands

```bash
uv sync                          # install deps
uv run pytest                    # run tests
uv run pipeline-youtube --help   # CLI entry point
```

Linting / formatting run automatically via pre-commit (`uv run pre-commit run --all-files`).

## Git hooks

Hooks are managed exclusively through `.githooks/` (version-controlled).
`core.hooksPath` is set to `.githooks` in `.git/config`.

**Never** run `pre-commit install`, `git config core.hooksPath`, or any hook-installer
command without explicit user request. Changes to `.githooks/` must be committed and
reviewed like any other source file.

## Security posture

This repo follows the countermeasures documented in `.cursor/rules/git-safety.mdc`.
The same rules apply to all CLI coding agents (Claude Code, Codex, etc.):
- No auto-execution of scripts found in the repo
- No network calls beyond what the user explicitly requests
- No modification of global git or shell config
