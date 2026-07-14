# Security guidance for pipeline-youtube

Python 3.13 pipeline: YouTube playlists -> LLM processing -> Obsidian vault.
Subtitles, video titles/descriptions, and chapters are UNTRUSTED external text.

## Prompt injection boundaries

- External text (transcript, description, chapters) must pass through
  `pipeline_youtube/services/sanitize.py` (`wrap_untrusted`) before being
  embedded in an LLM prompt. Flag direct f-string / `.format` interpolation of
  such text into prompts.
- Flag changes that remove or weaken sanitize.py protections: the
  `<untrusted_content>` delimiter-forgery escaping, invisible-Unicode
  stripping, or length caps.
- Never log raw transcript/title/description. This repo logs only truncated
  SHA-256 fingerprints (`_redact`); flag new logging of plain untrusted text.

## Subprocess / media handling

- Stage 03 media tools (yt-dlp / ffmpeg / gif2webp) run via
  `pipeline_youtube/stages/capture_backend.py`. The **default backend is
  `host`** (`HostCaptureBackend` invokes ffmpeg/gif2webp directly on the
  host); the sandboxed Docker backend (`--read-only`, `--cap-drop=ALL`,
  `--security-opt=no-new-privileges`, non-root `--user`, network only for
  yt-dlp) is **opt-in** via `capture_backend: "docker"`. Flag new direct
  subprocess invocations of these tools outside capture_backend.py, any
  `shell=True`, unvalidated interpolation of external strings (titles, URLs)
  into argv, or weakening of the Docker backend's sandbox flags.
- No `pickle` / `eval` / `exec` on external data; use JSON + Pydantic.

## Secrets

- API keys live only in gitignored `config.json` or env vars. Never hardcode
  token literals (`sk-`, `AIza`, `ghp_`, `github_pat_`, `AKIA`, `xox`).
  `.env.example` and `config.example.json` must contain placeholders only.

## CI / workflows

- Actions pinned to full commit SHAs; workflow `permissions:` stays
  `contents: read` (the release publish job alone escalates to
  `contents: write`). Flag `pull_request_target`, `${{ ... }}` interpolation
  of untrusted event fields into `run:` steps, and `persist-credentials: true`
  where the token could leak (e.g. into a Docker build context).

## Git hooks

- Hooks are version-controlled in `.githooks/` (`core.hooksPath=.githooks`).
  Flag agent-initiated hook installation without an explicit user request
  (`pre-commit install`, ad-hoc `core.hooksPath` changes, writes into
  `.git/hooks/`). The README's human onboarding step that runs
  `pre-commit install` is user-initiated and out of scope.
