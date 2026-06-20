# CLAUDE.md — pipeline-youtube-repo

This is the authoritative CLAUDE.md for this repository. It is version-controlled;
any modification is visible in `git diff` and requires review.

> 3 層設計: 普遍ルール (行動原則 / セキュリティ境界 / エスカレーション) は共通グローバル層
> `CLAUDE.global.md` (= `~/.claude/CLAUDE.md` 想定) にある。本ファイル = リポ固有ハードルール。
> 下記の Architecture invariant / Git hooks / Security posture はこのリポ固有のハードルールなのでここに残す。

## スキル発火表

このリポは現状**追加スキルなし** (`.claude/skills/` を持たない)。CLAUDE.md が既に薄く、
Git hooks / Security posture はハードルールとして常時ロードに置くのが適切なため。
将来、特定タスクでしか要らない作業規約・機能知識が増えたら `.claude/skills/` へ分離し
本表に発火条件を追加すること。

## Project

YouTube playlist → Obsidian vault learning pipeline written in Python 3.13.

## PR 分割規律 (★PR 作成前に必ず適用)

PR を作成する前に、変更内容を**性質別に分類**する。レビュー容易性のため
**性質が異なるものは束ねない**:

- **異なる実行経路 / 異なるレビュー観点は別 PR** にする。
- **live 実注入 (runtime wiring) と seam-only (準備) は別 PR** にする。
- **依存更新とアプリロジックを混在させない**。
- 束ね PR を作る場合は **Draft かつ umbrella と明記し、直接 merge の対象にしない**
  (個別の分割 PR を merge 対象とする)。

> 1 PR = 1 レビュー観点。チェックリストは `.github/pull_request_template.md` の
> Change Type / PR Scope Check を使う。

## Architecture invariant: main.py is a thin orchestrator

`main.py` は合成ルート (composition root)。残してよいのは
**CLI 定義 (引数/オプション)・段階の実行順序・モジュールの配線・終了/エラー処理**のみ。
グローバル CLAUDE.md が普遍ルールだけを持つのと同じ発想で、main.py には「普遍的な制御フロー」
だけを置き、各機能の HOW はモジュールへ出す (これが 2026 年の ~1372 行肥大化を招いた反省)。

- 機能の HOW (ロジック / パース / I/O / 分岐) は専用モジュールへ置き、main.py からは
  **呼び出す・配線する**だけにする (例: `cli_config.py` / `video_processing.py` /
  `run_result.py` / `resume.py` / `proper_noun_sheet.py`)。
- 切替・モードは `if/elif` の累積ではなく **config 値 + registry/strategy** で表現する
  (例: フォールバック chain の `fetchers=[("innertube", …), ("official", …), …]`、
  `use_innertube` のような config フラグ)。
- 1 機能で main.py に増えてよいのは原則「呼び出し or 配線 数行」。これを超える追加は、
  **先に対象モジュールへ抽出**してから行う。
- 目安: `main.py` ≤ ~500 行。超過が見込まれる変更は抽出を着手条件とする。
- リトマス試験: main.py を 2 分読んで「何が・どの順で・何に繋がって起きるか」が分かること。
  HOW が漏れていたら抽出のサイン。
- 関係図: `docs/main-architecture.md`（main.py の import / 配線を中心にした地図）。
  **main.py の import・配線・段階の順序を変えたら、同じ PR でこの図も更新する**（連動必須）。

> 新機能の着手前に「配置先モジュール」と「main.py への変更 = なし / 配線のみ (想定行数)」を
> 要件として宣言する。オーケストレータを編集せずに足せない設計は、まだ main.py 依存が残っている。

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
